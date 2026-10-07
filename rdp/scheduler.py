"""Run many instances in parallel, pausing for provider limits and resuming cleanly.

Each worker thread owns one instance at a time (its worktree, Claude session and prompter
chat). Workers share only the queue, the limit gate, and append-only logs.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from rdp.instances import Instance
from rdp.runner.experiment import RunContext, discard_run, run_instance
from rdp.runner.limit_detect import LimitHit, ProviderLimitError
from rdp.state import BLOCKED, DONE, FAILED, PENDING, RUNNING, load_state, log_event, save_state
from rdp.util import eprint, file_lock


@dataclass(frozen=True)
class PauseConfig:
    buffer_minutes: float = 2
    fixed_wait_minutes: float = 60
    max_wait_minutes: float = 480
    poll_seconds: float = 30


class LimitGate:
    """Shared pause. Once any worker reports a provider limit, no new work starts until it lifts."""

    def __init__(self, cfg: PauseConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._until: datetime | None = None

    def note(self, hit: LimitHit) -> datetime:
        now = datetime.now().astimezone()
        if hit.reset_dt is not None:
            target = hit.reset_dt.astimezone() + timedelta(minutes=self.cfg.buffer_minutes)
        else:
            target = now + timedelta(minutes=self.cfg.fixed_wait_minutes)
        target = min(max(target, now), now + timedelta(minutes=self.cfg.max_wait_minutes))
        with self._lock:
            if self._until is None or target > self._until:
                self._until = target
            return self._until

    def blocked_until(self) -> datetime | None:
        with self._lock:
            if self._until is not None and datetime.now().astimezone() >= self._until:
                self._until = None
            return self._until

    def wait(self, stop: threading.Event) -> None:
        announced = False
        while not stop.is_set():
            until = self.blocked_until()
            if until is None:
                return
            if not announced:
                eprint(f"[limit] paused until {until.strftime('%Y-%m-%d %H:%M %Z')}")
                announced = True
            remaining = (until - datetime.now().astimezone()).total_seconds()
            stop.wait(max(0.05, min(self.cfg.poll_seconds, remaining)))


def recover_stale(ctx: RunContext, instances: list[Instance]) -> int:
    """A `running` state with no live scheduler is a killed attempt: discard it."""
    n = 0
    for inst in instances:
        state = load_state(ctx.run_dir, inst.instance_id)
        if state.status in (RUNNING, BLOCKED):
            discard_run(ctx, inst, state, reason=f"stale_{state.status}")
            state.status = PENDING
            save_state(ctx.run_dir, state)
            n += 1
    return n


def run_all(
    ctx: RunContext,
    instances: list[Instance],
    *,
    jobs: int,
    pause: PauseConfig,
    max_requeues: int = 3,
    retry_failed: bool = False,
) -> dict[str, int]:
    """Run every instance that is not already done. Returns counts by final status."""
    with file_lock(ctx.run_dir / ".run.lock"):
        return _run_all_locked(ctx, instances, jobs=jobs, pause=pause,
                               max_requeues=max_requeues, retry_failed=retry_failed)


def _run_all_locked(ctx, instances, *, jobs, pause, max_requeues, retry_failed) -> dict[str, int]:
    stale = recover_stale(ctx, instances)
    if stale:
        eprint(f"[run] discarded {stale} interrupted attempt(s) from an earlier session")

    todo: list[Instance] = []
    for inst in instances:
        state = load_state(ctx.run_dir, inst.instance_id)
        if state.status == DONE or (state.status == FAILED and not retry_failed):
            continue
        todo.append(inst)

    work: queue.Queue[Instance] = queue.Queue()
    for inst in todo:
        work.put(inst)
    gate = LimitGate(pause)
    stop = threading.Event()
    eprint(f"[run] {len(todo)} instance(s) to run, {len(instances) - len(todo)} skipped, jobs={jobs}")

    def worker() -> None:
        while not stop.is_set():
            try:
                inst = work.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                gate.wait(stop)
                if stop.is_set():
                    continue
                state = load_state(ctx.run_dir, inst.instance_id)
                try:
                    state = run_instance(ctx, inst, state)
                    log_event(ctx.run_dir, "instance_finished", instance_id=inst.instance_id,
                              status=state.status, turns=state.turns_done, error=state.error)
                except ProviderLimitError as exc:
                    until = gate.note(exc.hit)
                    state.requeues += 1
                    log_event(ctx.run_dir, "provider_limit", instance_id=inst.instance_id,
                              kind=exc.hit.kind, raw=exc.hit.raw[:300], until=until.isoformat(),
                              requeues=state.requeues)
                    if state.requeues > max_requeues:
                        state.status = FAILED
                        state.error = f"discarded {state.requeues} times by provider limits"
                    else:
                        state.status = BLOCKED
                        work.put(inst)
                    save_state(ctx.run_dir, state)
            except Exception as exc:  # never let one instance kill a worker
                eprint(f"[run] {inst.instance_id}: unexpected error: {exc!r}")
                log_event(ctx.run_dir, "worker_error", instance_id=inst.instance_id, error=repr(exc))
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, name=f"rdp-worker-{i}", daemon=True)
               for i in range(max(1, jobs))]
    for t in threads:
        t.start()
    try:
        while work.unfinished_tasks:
            time.sleep(0.2)
    except KeyboardInterrupt:
        eprint("[run] interrupted: in-flight attempts will be discarded on the next `rdp run`")
        stop.set()
        raise
    stop.set()
    for t in threads:
        t.join(timeout=5)

    counts: dict[str, int] = {}
    for inst in instances:
        status = load_state(ctx.run_dir, inst.instance_id).status
        counts[status] = counts.get(status, 0) + 1
    return counts
