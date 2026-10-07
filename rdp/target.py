"""Name the target method unambiguously for turn 1.

Preferred label: `Class.method(ParamType, ...)`. If that cannot be derived or does not single
out one method in the file, fall back to the declaration quoted verbatim. If neither identifies
exactly one method, the instance has no usable label and is dropped at `prepare`.
"""

from __future__ import annotations

import re

_KEYWORD_TYPES = {"return", "new", "throw", "else", "case", "assert", "yield", "do", "await"}
_DECL_PREFIX = (
    r"(?m)^[ \t]*(?:(?:public|private|protected|static|final|abstract|synchronized|native|default|strictfp)\s+"
    r"|@\w[\w.]*(?:\([^)]*\))?\s+|<[^>;{}()]+>\s+)*([\w$][\w$<>\[\],.? ]*?)\s+"
)


def strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    return re.sub(r"//[^\n]*", " ", source)


def _close_paren(text: str, start: int) -> int:
    """Index just past the parenthesis that closes the one opened before `start`; -1 if none."""
    depth, j = 1, start
    while j < len(text) and depth:
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
        j += 1
    return j if depth == 0 else -1


def _split_params(params: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in params:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def _param_type(param: str) -> str | None:
    param = re.sub(r"@\w[\w.]*(\([^)]*\))?\s*", "", param).strip()
    param = re.sub(r"\bfinal\s+", "", param)
    m = re.match(r"(.+?)\s*(\.\.\.)?\s*[\w$]+\s*((?:\[\s*\])*)$", param, flags=re.S)
    if not m:
        return None
    return re.sub(r"\s+", "", m.group(1)) + ("..." if m.group(2) else "") + re.sub(r"\s+", "", m.group(3))


def _param_types(params: str) -> list[str] | None:
    types = [_param_type(p) for p in _split_params(params)]
    return None if any(t is None for t in types) else types


def method_header(method_source: str, name: str) -> tuple[list[str] | None, str]:
    """(parameter types or None, declaration text up to the closing parenthesis) of a method's source."""
    src = strip_comments(method_source)
    m = re.search(r"(?<![\w$.])" + re.escape(name) + r"\s*\(", src)
    if not m:
        return None, ""
    end = _close_paren(src, m.end())
    if end < 0:
        return None, ""
    # Drop annotations that precede the declaration; keep the parameter list exactly as written.
    before = re.sub(r"@\w[\w.]*(\([^)]*\))?\s*", "", src[:m.start()])
    declaration = re.sub(r"\s+", " ", before + src[m.start():end]).strip()
    return _param_types(src[m.end():end - 1]), declaration


def count_declarations(file_source: str, name: str, types: list[str]) -> int:
    """How many methods named `name` with exactly these parameter types the file declares."""
    src = strip_comments(file_source)
    n = 0
    for m in re.finditer(_DECL_PREFIX + re.escape(name) + r"\s*\(", src):
        if m.group(1).strip().split()[-1] in _KEYWORD_TYPES:
            continue
        end = _close_paren(src, m.end())
        if end < 0:
            continue
        tail = src[end:end + 200].lstrip()
        if not (tail.startswith("{") or tail.startswith("throws") or tail.startswith(";")):
            continue
        if _param_types(src[m.end():end - 1]) == types:
            n += 1
    return n


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def target_label(qualified: str, method_source: str, file_source: str) -> tuple[str, str]:
    """(label, form). form is 'signature', 'declaration', or '' when no label identifies one method."""
    if "#" not in qualified:
        return "", ""
    class_part, name = qualified.rsplit("#", 1)
    class_name = class_part.split(".")[-1].replace("$", ".")
    types, declaration = method_header(method_source, name)
    if types is not None and count_declarations(file_source, name, types) == 1:
        return f"{class_name}.{name}({', '.join(types)})", "signature"
    if declaration and _squash(strip_comments(file_source)).count(_squash(declaration)) == 1:
        return declaration, "declaration"
    return "", ""
