# notebooks/datagen

Phase 1. Builds a fingerprinted, deduplicated, leakage-safe dataset from Hugging Face sources only, plus a locked adversarial set and the LoRA's held-out comparison set.

- `phase_1_prompt_injection_datagen_v4.ipynb`: current. Twenty sources across chat, document and agentic shapes; every clean-and-injected pair shares a group key; all nine reasons populated. Its header cell lists every source with licence, rows, label, reason and shape, and every rejected source with the reason.
- `phase_1_prompt_injection_datagen_v3.ipynb`: kept for the record. Produced the v3 data the v3 verdict in the README was measured on.

Run top to bottom. The last cell must print `leakage checks passed` and writes `reports/<stamp>_phase_1_datagen_<version>_<fingerprint>/report.md`. Earlier versions' output folders are never overwritten. See `docs/PROCESS.md` section 4.
