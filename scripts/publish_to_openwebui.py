#!/usr/bin/env python3
"""Publish the package and a pair of model runs into an Open WebUI instance's data directory.

This is a convenience for people who train their own runs (or who prefer a script). Anyone else
installs the downloadable bundle by hand; see filter/README.md, section "Install". The DEFAULT_TARGET
below is the author's host; everyone else passes --target.

    python3 scripts/publish_to_openwebui.py --target /path/to/open-webui/data/prompt-injection-nn
    python3 scripts/publish_to_openwebui.py --target ... --from-bundle prompt-injection-nn-models-v0.2.0-<fp>.tar.gz
    python3 scripts/publish_to_openwebui.py --target ... --phase3 phase3-cnn-bigru-v4-<fp> --phase2 phase2-v4-<fp>

Default target: the Open WebUI data directory's prompt-injection-nn/ folder, which the container already
mounts and which the filter creates on first load; nothing is added to the stack. --target lets an operator
use a separate folder (then mount it read-only and set the two path valves).

What it does, idempotently:
  1. copies src/prompt_injection_nn/ into <target>/src/ (replacing the previous copy),
  2. picks the newest phase3-cnn-bigru-* run and the newest phase2-* run under models/ (or the ones
     named), checks that both carry the same dataset fingerprint, and copies them into <target>/models/,
  3. removes other run folders from <target>/models/ unless --keep-old is given, so the package's
     "newest run" resolution inside the container always finds the published pair,
  4. writes <target>/published.json (what was published, from where, when, by which commit),
  5. prints the compose volume line and the filter valve values.

Nothing here touches the container. Stdlib only; runs as the host user; needs no sudo because the
docker-data folder belongs to the host user and the run folders are world-readable.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The filter's default home is <Open WebUI data dir>/prompt-injection-nn, which the container sees as
# /app/backend/data/prompt-injection-nn and which the filter creates on first load. On this host the
# data dir is bind-mounted from /home/spark/docker-data/openwebui/data, so nothing is added to the stack.
DEFAULT_TARGET = "/home/spark/docker-data/openwebui/data/prompt-injection-nn"
CONTAINER_MOUNT = "/app/backend/data/prompt-injection-nn"
REQUIRED = {
    "phase3": ("model_state.pt", "vocab.json", "tokenizer.json", "config.json", "thresholds.json", "gate.json", "run_manifest.json"),
    "phase2": ("tfidf_word.joblib", "tfidf_char.joblib", "logreg.joblib", "logreg_calibrator.joblib", "thresholds.json", "run_manifest.json"),
}


def newest(models_dir: Path, prefix: str):
    runs = sorted((p for p in models_dir.glob(f"{prefix}*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def fingerprint(run: Path) -> str:
    return json.loads((run / "run_manifest.json").read_text())["phase1_config_fingerprint"]


def copy_tree(src: Path, dst: Path):
    """Replace dst with a copy of src. The previous dst is moved aside first (a rename only needs rights
    on the parent folder), then deleted best-effort: files the container created as root, such as
    __pycache__, cannot be removed by the host user and are left in a .retired-* folder, reported at the end."""
    tmp = dst.with_name(dst.name + ".publishing")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints", ".cache"))
    if dst.exists():
        retired = dst.with_name(f"{dst.name}.retired-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.rename(dst, retired)
        shutil.rmtree(retired, ignore_errors=True)
        if retired.exists():
            LEFTOVERS.append(retired)
    os.replace(tmp, dst)


LEFTOVERS = []


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"host folder the container mounts read-only (default {DEFAULT_TARGET})")
    parser.add_argument("--models", default=str(ROOT / "models"), help="models directory to publish from (default: the repo's models/)")
    parser.add_argument("--phase3", default=None, help="phase3-cnn-bigru-* run folder name (default: newest)")
    parser.add_argument("--phase2", default=None, help="phase2-* run folder name (default: newest)")
    parser.add_argument("--keep-old", action="store_true", help="leave previously published run folders in place")
    parser.add_argument("--no-runs", action="store_true", help="publish the package and the guard only: no phase2-*/phase3-* run folders (and remove any previously published ones unless --keep-old); for a deployment that uses the hf_guard_http backend")
    parser.add_argument("--from-bundle", default=None, help="a release tarball from scripts/package_release.py; extracted into --target instead of copying from --models")
    args = parser.parse_args()

    if args.from_bundle:
        import tarfile
        target = Path(args.target)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.from_bundle) as tar:
            names = tar.getnames()
            assert all(n.startswith(("src/", "models/", "published.json")) for n in names), "unexpected paths in the bundle"
            tar.extractall(target, filter="data")
        print(f"extracted {args.from_bundle} into {target}")
        print("filter valves: package_path = <container data dir>/prompt-injection-nn/src, models_dir = .../models (the defaults)")
        return 0

    models = Path(args.models)
    p3 = p2 = None
    f3 = None
    gate = {"recommended_backend": "hf_guard_http"}
    if not args.no_runs:
        p3 = models / args.phase3 if args.phase3 else newest(models, "phase3-cnn-bigru-")
        p2 = models / args.phase2 if args.phase2 else newest(models, "phase2-")
        if p3 is None or p2 is None or not p3.is_dir() or not p2.is_dir():
            print(f"error: need a phase3-cnn-bigru-* and a phase2-* run under {models} (or pass --no-runs)", file=sys.stderr)
            return 2
        for run, kind in ((p3, "phase3"), (p2, "phase2")):
            missing = [f for f in REQUIRED[kind] if not (run / f).exists()]
            if missing:
                print(f"error: {run.name} is missing {missing}; rerun its notebook", file=sys.stderr)
                return 2
        f3, f2 = fingerprint(p3), fingerprint(p2)
        if f3 != f2:
            print(f"error: {p3.name} ({f3[:16]}) and {p2.name} ({f2[:16]}) were trained on different data builds", file=sys.stderr)
            return 2
        gate = json.loads((p3 / "gate.json").read_text())
    if (models / "deploy.json").exists():
        gate = {**gate, "recommended_backend": json.loads((models / "deploy.json").read_text()).get("backend", gate.get("recommended_backend"))}

    target = Path(args.target)
    (target / "models").mkdir(parents=True, exist_ok=True)
    copy_tree(ROOT / "src" / "prompt_injection_nn", target / "src" / "prompt_injection_nn")
    if p3 is not None and p2 is not None:
        copy_tree(p3, target / "models" / p3.name)
        copy_tree(p2, target / "models" / p2.name)
    # The transformer guard (models/hf/<name>/) and the deployment choice (models/deploy.json), when present.
    deploy_path = models / "deploy.json"
    hf_name = json.loads(deploy_path.read_text()).get("hf_model") if deploy_path.exists() else None
    for hf_dir in sorted((models / "hf").glob("*")) if (models / "hf").is_dir() else []:
        if hf_dir.is_dir() and (hf_dir / "config.json").exists() and (hf_name is None or hf_dir.name == hf_name):
            copy_tree(hf_dir, target / "models" / "hf" / hf_dir.name)
            print(f"  guard   : hf/{hf_dir.name}")
    if deploy_path.exists():
        shutil.copyfile(deploy_path, target / "models" / "deploy.json")
        print(f"  deploy  : {json.loads(deploy_path.read_text())}")
    removed = []
    if not args.keep_old:
        for old in (target / "models").iterdir():
            keep = {p.name for p in (p3, p2) if p is not None}
            if old.is_dir() and old.name not in keep and (old.name.startswith("phase2-") or old.name.startswith("phase3-")):
                shutil.rmtree(old)
                removed.append(old.name)

    published = {
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_commit(),
        "source_models_dir": str(models),
        "phase3_run": p3.name if p3 else None, "phase2_run": p2.name if p2 else None,
        "dataset_fingerprint": f3,
        "recommended_backend": gate.get("recommended_backend"),
        "container_mount": CONTAINER_MOUNT,
        "valves": {"package_path": f"{CONTAINER_MOUNT}/src", "models_dir": f"{CONTAINER_MOUNT}/models"},
    }
    (target / "published.json").write_text(json.dumps(published, indent=2) + "\n")

    if LEFTOVERS:
        print("note: these old folders hold files owned by the container's user and could not be deleted from the host;")
        print("      they are harmless, and `docker exec <open-webui container> rm -rf <container path>` removes them:")
        for path in LEFTOVERS:
            print(f"      {path}")
    size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file() and not any(part.startswith(("src.retired", "phase2-", "phase3-")) and ".retired-" in part for part in f.parts)) / 1e6
    print(f"published to {target}  ({size_mb:.0f} MB)")
    print(f"  package : src/prompt_injection_nn  (commit {published['repo_commit']})")
    print(f"  phase 3 : {p3.name if p3 else '(none, --no-runs)'}")
    print(f"  phase 2 : {p2.name if p2 else '(none, --no-runs)'}")
    print(f"  data    : {f3[:16] if f3 else '-'}   recommended backend: {published['recommended_backend']}")
    if removed:
        print(f"  removed : {', '.join(removed)}")
    print()
    if str(target) == DEFAULT_TARGET:
        print("no stack change needed: this folder is inside the data directory the container already mounts,")
        print(f"and the filter's default valves point at {CONTAINER_MOUNT}.")
    else:
        print("non-default target: mount it into the container (read-only is fine) and set the valves to match:")
        print(f"      - {target}:{CONTAINER_MOUNT}:ro")
    print()
    print("filter valves (the defaults in the filter file):")
    print(f"  package_path = {published['valves']['package_path']}")
    print(f"  models_dir   = {published['valves']['models_dir']}")
    print()
    print("a later model update: run this script again; no container restart (the newest run folders are loaded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
