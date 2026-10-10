#!/usr/bin/env python3
"""Judge a transformer guard on our locked v4 rows with the same gate and budgets as the in-house models.

Reads probabilities_{val,test,adversarial_test}.parquet written by scripts/score_hf_guard.py beside the model,
and the Phase 3 run's probabilities for the baseline and in-house candidates on the identical rows. Writes beside
the model: thresholds.json (F1-max on val; FPR budgets on all val rows and on val document + agentic rows),
metrics.json (test, per shape, gate rows, adversarial, operating points), and, when the guard clears the gate,
models/deploy.json naming it as the backend to ship. Prints a comparison table.

    python3 scripts/evaluate_hf_guard.py --model models/hf/horizon-labs-prompt-injection-guard-base
"""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (0.01, 0.005, 0.001)
GATE_SHAPES = ("document", "agentic")


def recall_at_fpr(y, p, budgets=BUDGETS):
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64); neg = np.sort(p[y == 0]); pos = p[y == 1]; out = {}
    for b in budgets:
        allowed = int(np.floor(b * len(neg)))
        thr = float(np.nextafter(neg[-allowed - 1], 1.0)) if allowed < len(neg) else 0.0
        out[f"fpr_{b:g}"] = {"threshold": thr, "recall": float(np.mean(pos >= thr)) if len(pos) else None, "achieved_fpr": float(np.mean(neg >= thr)) if len(neg) else 0.0}
    return out


def f1_threshold(y, p):
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64); best = (-1, 0.5)
    for thr in np.unique(np.round(p, 4)):
        pred = p >= thr; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        if f1 > best[0]: best = (f1, float(thr))
    return best[1]


def at(y, p, thr):
    y = np.asarray(y); pred = np.asarray(p) >= thr
    return {"rows": int(len(y)), "recall": float(pred[y == 1].mean()) if (y == 1).any() else None, "fpr": float(pred[y == 0].mean()) if (y == 0).any() else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--column", default="hf_guard"); a = ap.parse_args()
    mdir = Path(a.model); models = ROOT / "models"
    p3 = sorted((d for d in models.glob("phase3-cnn-bigru-*") if d.is_dir()), key=lambda d: d.stat().st_mtime)[-1]
    guard = {s: pd.read_parquet(mdir / f"probabilities_{s}.parquet") for s in ("val", "test", "adversarial_test")}
    ours = {s: pd.read_parquet(p3 / f"probabilities_{s}.parquet") for s in ("val", "test", "adversarial_test")}
    for s in guard:
        assert (guard[s]["source_id"].astype(str).to_numpy() == ours[s]["source_id"].astype(str).to_numpy()).all(), f"{s}: rows not aligned"
    cands = {a.column: {s: guard[s][a.column].to_numpy() for s in guard}}
    for name in ("tfidf_logreg", "cnn_bigru", "ensemble_mean"):
        if name in ours["test"].columns: cands[name] = {s: ours[s][name].to_numpy() for s in ours}
    yv, yt = guard["val"]["label"].to_numpy(), guard["test"]["label"].to_numpy()
    shapes_v, shapes_t = guard["val"]["shape"].astype(str).to_numpy(), guard["test"]["shape"].astype(str).to_numpy()
    gv, gt = np.isin(shapes_v, GATE_SHAPES), np.isin(shapes_t, GATE_SHAPES)
    metrics = {}
    for name, p in cands.items():
        thr = f1_threshold(yv, p["val"]); val_all = recall_at_fpr(yv, p["val"]); val_ret = recall_at_fpr(yv[gv], p["val"][gv])
        m = {"threshold_f1_val": thr, "fpr_budget_thresholds_val": val_all, "fpr_budget_thresholds_val_retrieval": val_ret,
             "test": at(yt, p["test"], thr), "test_by_shape": {s: at(yt[shapes_t == s], p["test"][shapes_t == s], thr) for s in sorted(set(shapes_t))},
             "test_recall_at_fpr": {b: r["recall"] for b, r in recall_at_fpr(yt, p["test"]).items()},
             "test_recall_at_fpr_gate_rows": {b: r["recall"] for b, r in recall_at_fpr(yt[gt], p["test"][gt]).items()},
             "retrieval_operating_points": {b: {"threshold": r["threshold"], "test_gate_rows": at(yt[gt], p["test"][gt], r["threshold"]), "test_all": at(yt, p["test"], r["threshold"])} for b, r in val_ret.items()},
             "adversarial_recall": float(np.mean(p["adversarial_test"] >= thr)),
             "adversarial_recall_at_retrieval_fpr_0.01": float(np.mean(p["adversarial_test"] >= val_ret["fpr_0.01"]["threshold"]))}
        metrics[name] = m
    base = metrics["tfidf_logreg"]; verdicts = {}
    for name, m in metrics.items():
        if name == "tfidf_logreg": continue
        ok = all(m["test_recall_at_fpr_gate_rows"][b] >= base["test_recall_at_fpr_gate_rows"][b] for b in base["test_recall_at_fpr_gate_rows"])
        adv_ok = m["adversarial_recall"] >= base["adversarial_recall"] - 0.01
        verdicts[name] = {"clears_gate": bool(ok and adv_ok), "adversarial_ok": bool(adv_ok)}
    clearing = [n for n, v in verdicts.items() if v["clears_gate"]]
    best = max(clearing, key=lambda n: metrics[n]["test_recall_at_fpr_gate_rows"]["fpr_0.01"]) if clearing else "tfidf_logreg"
    print(f"{'candidate':14s} gate@1%  gate@0.5% gate@0.1%  | retrieval op @1%: recall / fpr | adv   | doc FPR@f1 | clears")
    for name, m in metrics.items():
        g = m["test_recall_at_fpr_gate_rows"]; op = m["retrieval_operating_points"]["fpr_0.01"]["test_gate_rows"]
        print(f"{name:14s} {g['fpr_0.01']:.3f}    {g['fpr_0.005']:.3f}     {g['fpr_0.001']:.3f}      | {op['recall']:.3f} / {op['fpr']:.4f}             | {m['adversarial_recall']:.3f} | {m['test_by_shape']['document']['fpr']:.4f}     | {verdicts.get(name, {}).get('clears_gate', '-')}")
    print("recommended backend:", best)
    gm = metrics[a.column]
    (mdir / "thresholds.json").write_text(json.dumps({"threshold": gm["threshold_f1_val"], "selected_on": "v4 val.parquet, max F1", "fpr_budget_thresholds_val": gm["fpr_budget_thresholds_val"],
                                                       "fpr_budget_thresholds_val_retrieval": gm["fpr_budget_thresholds_val_retrieval"]}, indent=2, sort_keys=True) + "\n")
    (mdir / "metrics.json").write_text(json.dumps({"evaluated_utc": datetime.now(timezone.utc).isoformat(), "phase3_run": p3.name, "candidates": metrics, "gate_verdicts": verdicts, "recommended_backend": best}, indent=2, sort_keys=True, default=str) + "\n")
    (mdir / "run_manifest.json").write_text(json.dumps({"model": mdir.name, "source": "Horizon-Labs/prompt-injection-guard-base" if "horizon" in mdir.name else mdir.name, "license": "apache-2.0",
                                                         "evaluated_on": "v4 locked splits, CPU, windowed max (2048/512)", "evaluated_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    if best == a.column:
        (models / "deploy.json").write_text(json.dumps({"backend": "hf_guard", "hf_model": mdir.name, "chosen_utc": datetime.now(timezone.utc).isoformat(),
                                                         "why": "clears the gate on test document + agentic rows with the highest recall at the 1% budget"}, indent=2) + "\n")
        print("wrote models/deploy.json -> hf_guard")
    print("wrote", mdir / "thresholds.json", "and", mdir / "metrics.json")


if __name__ == "__main__":
    sys.exit(main())
