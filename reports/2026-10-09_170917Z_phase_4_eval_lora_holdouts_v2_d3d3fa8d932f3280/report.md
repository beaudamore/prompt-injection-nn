# Phase 4 Report: Evaluation on the LoRA held-out sets (phase3-cnn-bigru-v2-d3d3fa8d932f3280)

_Generated 2026-10-09 17:09 UTC_

## Run

- Phase 3 run: `models/phase3-cnn-bigru-v2-d3d3fa8d932f3280`
- Dataset fingerprint: `d3d3fa8d932f3280ecfb9e63a0ca7fa4051f80af25944e2fd2ab09d871a582f8`
- Threshold: 0.4330
- Held-out rows: 2,086 (wildguard_test 1,286, nemotron_test_jailbreaking 800)
- Output: `eval_wildguard_test.json` and `eval_wildguard_test_rows.jsonl` in the run directory

## LoRA-format summary

- Accuracy (SAFE/INJECTION): 0.901
- Invalid output: 0 (0.0%)
- False positives (SAFE -> INJECTION): 89 / 945 = 9.4%
- False negatives (INJECTION -> SAFE): 118 / 1141 = 10.3%

|  | rows | correct | accuracy |
|---|---|---|---|
| nemotron_test_jailbreaking | 800.000 | 776.000 | 0.970 |
| wildguard_test | 1286.000 | 1103.000 | 0.858 |

## Probability-based metrics

|  | threshold | precision | recall | f1 | roc_auc | average_precision | brier | log_loss | false_positive_rate |
|---|---|---|---|---|---|---|---|---|---|
| lora_holdout | 0.4330 | 0.9200 | 0.8966 | 0.9081 | 0.9642 | 0.9689 | 0.0733 | 0.2467 | 0.0942 |

95% bootstrap intervals (1,000 resamples):

|  | average_precision | roc_auc | f1 | recall | false_positive_rate |
|---|---|---|---|---|---|
| lora_holdout | 0.9605 to 0.9766 | 0.9569 to 0.9715 | 0.8968 to 0.9199 | 0.8793 to 0.9135 | 0.0762 to 0.1129 |

## Recall at FPR budget

|  | recall | threshold | achieved_fpr |
|---|---|---|---|
| fpr_0.01 | 0.5259 | 0.9859 | 0.0095 |
| fpr_0.005 | 0.3742 | 0.9968 | 0.0042 |
| fpr_0.001 | 0.2226 | 0.9992 | 0.0000 |

## Slices by source

|  | rows | recall | false_positive_rate | f1 |
|---|---|---|---|---|
| nemotron_test_jailbreaking | 800.0000 | 0.9700 |  |  |
| wildguard_test | 1286.0000 | 0.7243 | 0.0942 | 0.7297 |

## Predicted reasons on flagged rows

|  | predicted |
|---|---|
| Privilege Escalation | 50 |
| Override Attempt | 4 |
| Jailbreak Technique | 1055 |
| Embedded Instruction | 1 |
| Meta-Command Injection | 2 |

## Comparison status

No LoRA `eval_*.json` exists yet in the `safety` repo; this report is one side of the head-to-head. The LoRA eval notebook writes the same `rows`, `accuracy`, `invalid`, `false_positives`, `safe_rows`, `false_negatives`, `injection_rows` and `sources` fields.
