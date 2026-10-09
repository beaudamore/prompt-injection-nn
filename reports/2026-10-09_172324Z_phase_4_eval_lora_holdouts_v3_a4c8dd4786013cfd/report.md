# Phase 4 Report: Evaluation on the LoRA held-out sets (phase3-cnn-bigru-v3-a4c8dd4786013cfd)

_Generated 2026-10-09 17:23 UTC_

## Run

- Phase 3 run: `models/phase3-cnn-bigru-v3-a4c8dd4786013cfd`
- Dataset fingerprint: `a4c8dd4786013cfddd09959eb76105808ccf986687152acd19a0a998d1f17811`
- Threshold: 0.5148
- Held-out rows: 2,086 (wildguard_test 1,286, nemotron_test_jailbreaking 800)
- Output: `eval_wildguard_test.json` and `eval_wildguard_test_rows.jsonl` in the run directory

## LoRA-format summary

- Accuracy (SAFE/INJECTION): 0.905
- Invalid output: 0 (0.0%)
- False positives (SAFE -> INJECTION): 117 / 945 = 12.4%
- False negatives (INJECTION -> SAFE): 81 / 1141 = 7.1%

|  | rows | correct | accuracy |
|---|---|---|---|
| nemotron_test_jailbreaking | 800.000 | 786.000 | 0.983 |
| wildguard_test | 1286.000 | 1102.000 | 0.857 |

## Probability-based metrics

|  | threshold | precision | recall | f1 | roc_auc | average_precision | brier | log_loss | false_positive_rate |
|---|---|---|---|---|---|---|---|---|---|
| lora_holdout | 0.5148 | 0.9006 | 0.9290 | 0.9146 | 0.9669 | 0.9726 | 0.0730 | 0.2446 | 0.1238 |

95% bootstrap intervals (1,000 resamples):

|  | average_precision | roc_auc | f1 | recall | false_positive_rate |
|---|---|---|---|---|---|
| lora_holdout | 0.9655 to 0.9785 | 0.9596 to 0.9731 | 0.9019 to 0.9263 | 0.9137 to 0.9422 | 0.1032 to 0.1454 |

## Recall at FPR budget

|  | recall | threshold | achieved_fpr |
|---|---|---|---|
| fpr_0.01 | 0.6161 | 0.9855 | 0.0095 |
| fpr_0.005 | 0.5364 | 0.9931 | 0.0042 |
| fpr_0.001 | 0.3006 | 0.9993 | 0.0000 |

## Slices by source

|  | rows | recall | false_positive_rate | f1 |
|---|---|---|---|---|
| nemotron_test_jailbreaking | 800.0000 | 0.9825 |  |  |
| wildguard_test | 1286.0000 | 0.8035 | 0.1238 | 0.7486 |

## Predicted reasons on flagged rows

|  | predicted |
|---|---|
| Jailbreak Technique | 1143 |
| Privilege Escalation | 26 |
| Override Attempt | 7 |
| Embedded Instruction | 1 |

## Comparison status

No LoRA `eval_*.json` exists yet in the `safety` repo; this report is one side of the head-to-head. The LoRA eval notebook writes the same `rows`, `accuracy`, `invalid`, `false_positives`, `safe_rows`, `false_negatives`, `injection_rows` and `sources` fields.
