"""Small helpers for safe notebook edits. See SKILL.md.

Usage:
  python -I nb_tools.py dump <notebook> <cell_index>
  python -I nb_tools.py outline <notebook>
  python -I nb_tools.py verify <notebook> [<notebook> ...]
"""
import ast
import json
import re
import sys
from pathlib import Path

MAGIC_RE = re.compile(r"^\s*[%!].*$", re.M)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, nb):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def dump(path, index):
    nb = load(path)
    cell = nb["cells"][int(index)]
    print(f"# {path} cell {index} ({cell['cell_type']}, exec={cell.get('execution_count')})")
    print("".join(cell["source"]))


def outline(path):
    nb = load(path)
    print(f"# {path}: nbformat {nb['nbformat']}.{nb['nbformat_minor']}, {len(nb['cells'])} cells")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"]).strip()
        first = src.splitlines()[0][:90] if src else ""
        ex = c.get("execution_count") if c["cell_type"] == "code" else "-"
        outs = len(c.get("outputs", [])) if c["cell_type"] == "code" else "-"
        print(f"{i:3d} {c['cell_type']:8s} exec={str(ex):4s} outputs={str(outs):3s} | {first}")


def verify(paths):
    ok = True
    for path in paths:
        raw = Path(path).read_bytes()
        crlf = raw.count(b"\r\n")
        if crlf:
            Path(path).write_bytes(raw.replace(b"\r\n", b"\n"))
            print(f"{path}: normalized {crlf} CRLF line endings to LF")
        nb = load(path)
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            src = MAGIC_RE.sub("pass", "".join(c["source"]))
            try:
                ast.parse(src, filename=f"{path}#cell{i}")
            except SyntaxError as exc:
                ok = False
                print(f"{path} cell {i}: SYNTAX ERROR {exc}")
        print(f"{path}: {len(nb['cells'])} cells, all code cells parse" if ok else f"{path}: FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "dump" and len(sys.argv) == 4:
        dump(sys.argv[2], sys.argv[3])
    elif cmd == "outline" and len(sys.argv) == 3:
        outline(sys.argv[2])
    elif cmd == "verify" and len(sys.argv) >= 3:
        verify(sys.argv[2:])
    else:
        print(__doc__)
        sys.exit(2)
