# notebooks/training

- `phase_2_baselines.ipynb`: TF-IDF logistic regression and XGBoost on engineered features. Calibrated on `val`, thresholds by validation F1, then the locked test, the adversarial set, recall at FPR budgets, bootstrap intervals and slices. The winner is the bar Phase 3 must clear.
- `phase_3_text_cnn_bigru.ipynb`: CNN + bidirectional GRU with a binary head and a nine-way reason head. Same evaluation, plus the reason-head table and CPU latency.

Both read `DATASET_VERSION` from their configuration cell (currently `v4`), assert the manifest and fingerprint before training, and write `models/<run-id>/` and a report folder under `reports/`. The promotion gate is in `docs/PLAN.md` section 4. See `docs/PROCESS.md` section 4.
