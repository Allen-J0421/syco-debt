#!/usr/bin/env python3
"""aggregate.py — derive the six code-quality metrics for ONE snapshot.

Inputs: a CK output dir (class.csv + method.csv) and a SonarQube measures JSON.
Output: a flat dict of metric fields (no identity columns — callers add those).

  CLI:  aggregate.py <ck_dir> <sonar_json> <out_metric_json>

Metric sources:
  Cyclomatic Complexity      Sonar `complexity` (total, /function) + CK method `wmc`
  Cognitive Complexity       Sonar `cognitive_complexity` (total, /function)
  Code Smell Count           Sonar `code_smells` (count, /kLOC)
  Duplication Ratio          Sonar `duplicated_lines_density`
  Coupling Between Objects   CK class-level `cbo` (mean / median / max)
  Maintainability            Sonar SQALE: rating A-E + technical-debt ratio
"""
import csv, json, os, statistics, sys

RATING = {"1.0": "A", "2.0": "B", "3.0": "C", "4.0": "D", "5.0": "E"}
# Column order shared by experiments + baselines metric block
METRIC_COLS = [
    "files", "ncloc",
    "cc_total", "cc_per_fn", "cc_mean_method_ck", "cc_max_method_ck",
    "cognitive_total", "cognitive_per_fn",
    "code_smells", "smells_per_kloc",
    "duplication_pct",
    "cbo_mean", "cbo_median", "cbo_max", "ck_classes",
    "maint_rating", "tech_debt_ratio_pct", "tech_debt_min",
]

def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def ck_metrics(ck_dir):
    out = {"ck_classes": 0, "cbo_mean": None, "cbo_median": None, "cbo_max": None,
           "cc_mean_method_ck": None, "cc_max_method_ck": None}
    cpath = os.path.join(ck_dir, "class.csv"); mpath = os.path.join(ck_dir, "method.csv")
    if os.path.exists(cpath):
        cbos = []
        with open(cpath, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                v = fnum(row.get("cbo"))
                if v is not None: cbos.append(v)
        if cbos:
            out["ck_classes"] = len(cbos)
            out["cbo_mean"] = round(statistics.mean(cbos), 2)
            out["cbo_median"] = round(statistics.median(cbos), 2)
            out["cbo_max"] = int(max(cbos))
    if os.path.exists(mpath):
        ccs = []
        with open(mpath, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                v = fnum(row.get("wmc"))
                if v is not None: ccs.append(v)
        if ccs:
            out["cc_mean_method_ck"] = round(statistics.mean(ccs), 2)
            out["cc_max_method_ck"] = int(max(ccs))
    return out

def sonar_metrics(json_path):
    d = {}
    if json_path and os.path.exists(json_path):
        try:
            j = json.load(open(json_path))
            for m in j.get("component", {}).get("measures", []):
                d[m["metric"]] = m.get("value")
        except Exception:
            pass
    return d

def compute(ck_dir, sonar_json):
    ck = ck_metrics(ck_dir)
    s = sonar_metrics(sonar_json)
    ncloc = fnum(s.get("ncloc")); funcs = fnum(s.get("functions"))
    cx = fnum(s.get("complexity")); cog = fnum(s.get("cognitive_complexity"))
    smells = fnum(s.get("code_smells"))
    return {
        "files": s.get("files"), "ncloc": s.get("ncloc"),
        "cc_total": s.get("complexity"),
        "cc_per_fn": round(cx/funcs, 2) if cx is not None and funcs else None,
        "cc_mean_method_ck": ck["cc_mean_method_ck"], "cc_max_method_ck": ck["cc_max_method_ck"],
        "cognitive_total": s.get("cognitive_complexity"),
        "cognitive_per_fn": round(cog/funcs, 2) if cog is not None and funcs else None,
        "code_smells": s.get("code_smells"),
        "smells_per_kloc": round(smells/ncloc*1000, 2) if smells is not None and ncloc else None,
        "duplication_pct": s.get("duplicated_lines_density"),
        "cbo_mean": ck["cbo_mean"], "cbo_median": ck["cbo_median"], "cbo_max": ck["cbo_max"],
        "ck_classes": ck["ck_classes"],
        "maint_rating": RATING.get(s.get("sqale_rating"), s.get("sqale_rating")),
        "tech_debt_ratio_pct": s.get("sqale_debt_ratio"), "tech_debt_min": s.get("sqale_index"),
    }

if __name__ == "__main__":
    ck_dir, sonar_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
    json.dump(compute(ck_dir, sonar_json), open(out, "w"))
