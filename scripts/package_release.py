#!/usr/bin/env python3
"""Build the downloadable model bundle for a GitHub Release.

    python3 scripts/package_release.py                 # newest matching runs under models/
    python3 scripts/package_release.py --version v0.2.0

Writes dist/prompt-injection-nn-models-<version>-<fingerprint16>.tar.gz and a .sha256 beside it.
The tarball extracts straight into the Open WebUI data directory:

    <open-webui data dir>/prompt-injection-nn/
        src/prompt_injection_nn/      the inference package
        models/phase2-*/              TF-IDF baseline run
        models/phase3-cnn-bigru-*/    CNN run (gate.json names the recommended backend)
        published.json                what this bundle is

Both run folders must carry the same dataset fingerprint. Stdlib only."""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from publish_to_openwebui import REQUIRED, fingerprint, newest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=None, help="release tag, e.g. v0.2.0 (default: the version in pyproject.toml)")
    parser.add_argument("--models", default=str(ROOT / "models"))
    parser.add_argument("--phase3", default=None)
    parser.add_argument("--phase2", default=None)
    parser.add_argument("--out", default=str(ROOT / "dist"))
    args = parser.parse_args()

    version = args.version
    if not version:
        for line in (ROOT / "pyproject.toml").read_text().splitlines():
            if line.startswith("version"):
                version = "v" + line.split('"')[1]
                break
    models = Path(args.models)
    p3 = models / args.phase3 if args.phase3 else newest(models, "phase3-cnn-bigru-")
    p2 = models / args.phase2 if args.phase2 else newest(models, "phase2-")
    if p3 is None or p2 is None:
        print(f"error: need a phase3-cnn-bigru-* and a phase2-* run under {models}", file=sys.stderr)
        return 2
    for run, kind in ((p3, "phase3"), (p2, "phase2")):
        missing = [f for f in REQUIRED[kind] if not (run / f).exists()]
        if missing:
            print(f"error: {run.name} is missing {missing}", file=sys.stderr)
            return 2
    f3, f2 = fingerprint(p3), fingerprint(p2)
    if f3 != f2:
        print(f"error: runs were trained on different data builds ({f3[:16]} vs {f2[:16]})", file=sys.stderr)
        return 2
    gate = json.loads((p3 / "gate.json").read_text())
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    published = {
        "bundle_version": version, "published_utc": datetime.now(timezone.utc).isoformat(), "repo_commit": commit,
        "phase3_run": p3.name, "phase2_run": p2.name, "dataset_fingerprint": f3,
        "recommended_backend": gate.get("recommended_backend"),
        "install": "extract into <open-webui data dir>/prompt-injection-nn/ so that src/ and models/ sit directly inside it",
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"prompt-injection-nn-models-{version}-{f3[:16]}.tar.gz"
    path = out / name
    skip = lambda ti: None if ("__pycache__" in ti.name or ti.name.endswith((".pyc", ".tmp")) or ti.name.endswith(("probabilities_val.parquet", "probabilities_test.parquet", "probabilities_adversarial_test.parquet", "eval_wildguard_test_rows.jsonl"))) else ti
    with tarfile.open(path, "w:gz") as tar:
        tar.add(ROOT / "src" / "prompt_injection_nn", arcname="src/prompt_injection_nn", filter=skip)
        tar.add(p3, arcname=f"models/{p3.name}", filter=skip)
        tar.add(p2, arcname=f"models/{p2.name}", filter=skip)
        info = tarfile.TarInfo("published.json")
        data = (json.dumps(published, indent=2) + "\n").encode()
        info.size = len(data); info.mtime = int(datetime.now().timestamp())
        import io
        tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / (name + ".sha256")).write_text(f"{digest}  {name}\n")
    print(f"built {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256 {digest}")
    print(f"  phase 3 {p3.name}\n  phase 2 {p2.name}\n  data    {f3[:16]}   recommended backend {published['recommended_backend']}")
    print()
    print("publish (from the repo, with the gh CLI authenticated):")
    print(f"  gh release create {version} {path} {out / (name + '.sha256')} --title 'Models {version}' --notes 'See docs/MODEL_CARD.md. Extract into <open-webui data dir>/prompt-injection-nn/.'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
