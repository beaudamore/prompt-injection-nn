---
name: pipeline-status
description: Snapshot where this training repo stands before doing anything else. Use when resuming work on any machine, when asked "where are we", or before editing a notebook.
---

# Pipeline status

Run every check below read-only, then summarize in five lines or fewer: git state, which data versions exist here, which model runs exist here, which notebooks have been executed, and what the next action is per `docs/PROCESS.md`.

## 1. Git state

```bash
git status --short --branch
git log -1 --format='%h %ci %s'
git fetch origin --quiet && git rev-list --left-right --count HEAD...@{u}
```

If behind, say so before anything else. Do not pull without being asked.

## 2. Data versions on this machine

```bash
ls -d data/source-clean* 2>/dev/null || echo "no data on this machine"
for d in data/source-clean*; do [ -f "$d/manifest.json" ] && python -I -c "import json,sys; m=json.load(open(sys.argv[1])); print(sys.argv[1], m.get('dataset_version','v1'), m['config_fingerprint'][:16], {k:v['rows'] for k,v in m['counts'].items()})" "$d/manifest.json"; done
```

Data is gitignored. Absence means this machine has not run Phase 1, not that the data is lost. The DGX Spark holds the canonical copies.

## 3. Model runs on this machine

```bash
ls models 2>/dev/null || echo "no models on this machine"
for d in models/*/; do [ -f "$d/run_manifest.json" ] && python -I -c "import json,sys; m=json.load(open(sys.argv[1])); print(sys.argv[1], m.get('dataset_version','v1'), m.get('chosen_model',''), m.get('created_utc',''))" "$d/run_manifest.json"; [ -f "$d/metrics.json" ] && [ ! -f "$d/run_manifest.json" ] && python -I -c "import json,sys; m=json.load(open(sys.argv[1])); print(sys.argv[1], m.get('dataset_version','v1'), m.get('created_utc',''))" "$d/metrics.json"; done
```

## 4. Notebook execution state

```bash
python -I -c "
import json,glob
for p in sorted(glob.glob('notebooks/**/*.ipynb', recursive=True)):
    nb=json.load(open(p,encoding='utf-8'))
    code=[c for c in nb['cells'] if c['cell_type']=='code']
    ex=sum(1 for c in code if c.get('execution_count'))
    err=sum(1 for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error')
    print(f'{p}: {ex}/{len(code)} code cells executed, {err} error outputs')
"
```

A notebook with fewer executed cells than code cells was interrupted. Say which cell it stopped at.

## 5. Next action

Read the "Where things stand" section of `CLAUDE.md` and section 4 of `docs/PROCESS.md`. State the single next step. Do not start it without being asked.

## Note for this repo

Until Phase 1 lands there are no notebooks, no data, and no models. "Nothing here yet" is the correct answer for sections 2 to 4, and the next action is Phase 1 per `CLAUDE.md`.
