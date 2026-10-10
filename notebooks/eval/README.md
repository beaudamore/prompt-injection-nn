# notebooks/eval

- `phase_4_eval_against_lora_holdouts.ipynb`: loads the newest Phase 3 run (or `RUN_ID`) and scores it on the LoRA's held-out sets (`lora_holdout.parquet`: WildGuardMix `wildguardtest` and Nemotron test-split `jailbreaking`). Writes `eval_wildguard_test.json` in the run directory with the same field names the `safety` repo's LoRA eval notebook uses, so the two detectors can be compared row for row once the LoRA eval has been run.

Trains nothing. Runs on CPU in a few minutes.
