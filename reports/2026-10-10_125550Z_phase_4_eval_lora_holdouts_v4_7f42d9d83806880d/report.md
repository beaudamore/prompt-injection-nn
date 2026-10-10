# Phase 4 Report: Evaluation on the LoRA held-out sets (phase3-cnn-bigru-v4-7f42d9d83806880d)

_Generated 2026-10-10 12:55 UTC_

## Run

- Phase 3 run: `models/phase3-cnn-bigru-v4-7f42d9d83806880d`
- Dataset fingerprint: `7f42d9d83806880d95a44645173227c6008b5291538e8af40eaae0ed37cf0486`
- Threshold: 0.3242
- Held-out rows: 2,086 (wildguard_test 1,286, nemotron_test_jailbreaking 800)
- Output: `eval_wildguard_test.json` and `eval_wildguard_test_rows.jsonl` in the run directory

## LoRA-format summary

- Accuracy (SAFE/INJECTION): 0.889
- Invalid output: 0 (0.0%)
- False positives (SAFE -> INJECTION): 126 / 945 = 13.3%
- False negatives (INJECTION -> SAFE): 105 / 1141 = 9.2%

|  | rows | correct | accuracy |
|---|---|---|---|
| nemotron_test_jailbreaking | 800.000 | 773.000 | 0.966 |
| wildguard_test | 1286.000 | 1082.000 | 0.841 |

## Probability-based metrics

|  | threshold | precision | recall | f1 | roc_auc | average_precision | brier | log_loss | false_positive_rate |
|---|---|---|---|---|---|---|---|---|---|
| lora_holdout | 0.3242 | 0.8916 | 0.9080 | 0.8997 | 0.9604 | 0.9678 | 0.0800 | 0.2678 | 0.1333 |

95% bootstrap intervals (1,000 resamples):

|  | average_precision | roc_auc | f1 | recall | false_positive_rate |
|---|---|---|---|---|---|
| lora_holdout | 0.9598 to 0.9743 | 0.9524 to 0.9671 | 0.8865 to 0.9120 | 0.8904 to 0.9239 | 0.1131 to 0.1556 |

## Recall at FPR budget

|  | recall | threshold | achieved_fpr |
|---|---|---|---|
| fpr_0.01 | 0.5776 | 0.9560 | 0.0095 |
| fpr_0.005 | 0.4698 | 0.9836 | 0.0042 |
| fpr_0.001 | 0.1840 | 0.9992 | 0.0000 |

## Slices by source

|  | rows | recall | false_positive_rate | f1 |
|---|---|---|---|---|
| nemotron_test_jailbreaking | 800.0000 | 0.9663 |  |  |
| wildguard_test | 1286.0000 | 0.7713 | 0.1333 | 0.7205 |

## Predicted reasons on flagged rows

|  | predicted |
|---|---|
| Jailbreak Technique | 1159 |
| Privilege Escalation | 3 |

## Comparison status

No LoRA `eval_*.json` exists yet in the `safety` repo; this report is one side of the head-to-head. The LoRA eval notebook writes the same `rows`, `accuracy`, `invalid`, `false_positives`, `safe_rows`, `false_negatives`, `injection_rows` and `sources` fields.
