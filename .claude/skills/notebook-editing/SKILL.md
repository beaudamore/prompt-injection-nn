---
name: notebook-editing
description: Rules and a helper for editing Jupyter notebooks in this repo without corrupting them or producing whole-file diffs. Use before any change to a .ipynb file.
---

# Notebook editing

Notebooks in this repo are committed **with outputs** and are the record of a run. Edit source cells surgically; never clear outputs; never reformat the file.

## Rules

1. **Dump the cell first.** Print the exact current source of every cell you intend to change. Do not edit from memory.
2. **Exact-match replacement, asserted.** Replace with `str.replace` only after asserting the old text occurs exactly once in that cell. A zero or multiple match is a stop, not a warning.
3. **LF line endings, `indent=1`, `ensure_ascii=False`, trailing newline.** On Windows, Python's `write_text` converts `\n` to `\r\n` and the diff touches every line. Write bytes, or normalize afterwards with the helper.
4. **Leave outputs and execution counts alone** unless the user asks to clear them.
5. **Verify.** After writing: every code cell must parse (neutralize `%` and `!` magics first), `git diff --stat` must be proportional to the change, and `git diff -w --stat` must be nearly the same as `git diff --stat`.
6. **Keep the notebook runnable on every machine.** Any new path or device logic goes through the repo's `resolve_project_dir()` / `resolve_device()` helpers already present in the v2 notebooks.
7. **State the change before making it.** Cell index, the old lines, the new lines.

## Helper

`nb_tools.py` in this directory:

```bash
# print a cell's source
python -I .claude/skills/notebook-editing/nb_tools.py dump notebooks/training/phase_2_baselines.ipynb 5

# list cells with type, exec count, first line
python -I .claude/skills/notebook-editing/nb_tools.py outline notebooks/training/phase_2_baselines.ipynb

# normalize CRLF -> LF in place and compile-check every code cell
python -I .claude/skills/notebook-editing/nb_tools.py verify notebooks/training/phase_2_baselines.ipynb
```

For the replacement itself, write a short script that loads the JSON, asserts the single match, replaces, and saves with `open(path, "w", encoding="utf-8", newline="\n")` and `json.dump(nb, f, indent=1, ensure_ascii=False)` followed by a trailing `"\n"`. Then run `verify`.
