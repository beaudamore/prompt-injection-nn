#!/usr/bin/env bash
# Export the executed notebooks to self-contained HTML under reports/html/<UTC stamp>/.
# Run after a notebook has finished and Jupyter has saved it (outputs included).
# The HTML is the full audit trail; the per-run report.md folders in reports/ are the curated summaries.
#
# Usage: scripts/export_html.sh [notebook ...]      (default: all four phase notebooks)
# Needs nbconvert, which ships with JupyterLab in the unsloth-notebook container.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="reports/html/$(date -u +%Y-%m-%d_%H%M%SZ)"
mkdir -p "$OUT"
if [ "$#" -eq 0 ]; then
  set -- notebooks/datagen/phase_1_prompt_injection_datagen.ipynb \
         notebooks/training/phase_2_baselines.ipynb \
         notebooks/training/phase_3_text_cnn_bigru.ipynb \
         notebooks/eval/phase_4_eval_against_lora_holdouts.ipynb
fi
python -m jupyter nbconvert --to html --output-dir "$OUT" "$@"
ls -l "$OUT"
