# Prompt Injection Classifier — Specialized Neural Network

A compact text classifier that flags prompt-injection content without calling an LLM. Built to
scan retrieved RAG chunks, web-search results, and uploaded files inside Open WebUI before they
reach the model, and to be importable from any Python app as a plain function.

Sibling of [email-classifier-nn](https://github.com/beaudamore/email-classifier-nn): same
pipeline discipline (Hugging Face sources only, fingerprinted dataset builds, leakage-safe splits,
baselines before the neural model, locked holdouts), same multi-device notebooks.

Replaces the per-call LLM round trip of the existing
[prompt-injection LoRA](https://github.com/beaudamore/safety) where latency matters. The LoRA
stays the detector for typed chat prompts; this model covers the retrieval path, where a single
turn can carry ten or more chunks.

**Status (2026-10-09):** scaffold only. No data built, nothing trained. See `docs/PLAN.md` for
the design and `docs/PROCESS.md` for how a run will work.

---

## What it will produce

| Artifact | Purpose |
|---|---|
| `src/` package | `classify(text)` and `classify_batch(texts)` returning a probability, a verdict, and a reason from the nine-reason taxonomy shared with the Open WebUI filter. |
| `filter/` | An Open WebUI `request`-hook filter that scans every `<source>` block the retrieval pipeline injected, drops flagged chunks, and removes them from the citation list. No Open WebUI patch required. |
| `models/<run-id>/` | Versioned weights, vocab, thresholds, metrics, and a run manifest tied to the dataset fingerprint. Gitignored; rebuilt by running the notebooks. |

## Output contract

Matches the existing filter and LoRA so the two detectors are interchangeable at the call site:

```text
SAFE
INJECTION: <reason>
```

Valid reasons: Override Attempt, Fake System Injection, Meta-Command Injection, Jailbreak
Technique, Delimiter Confusion, Privilege Escalation, Embedded Instruction, Role-Play Bypass,
Chain-of-Instructions.

## Repo layout

```text
prompt-injection-nn/
├── README.md
├── CLAUDE.md          Handoff context for Claude Code on any machine
├── .claude/skills/    pipeline-status, notebook-editing
├── docs/
│   ├── PLAN.md        Scope, data strategy, architecture, Open WebUI integration, phase roadmap
│   ├── PROCESS.md     How to run the pipeline on the DGX Spark, a Mac, or CPU
│   └── EVALUATION.md  Verification protocol: leakage audit, metrics, CIs, robustness, filter tests, human review
├── notebooks/
│   ├── datagen/       Phase 1: build fingerprinted splits from Hugging Face sources
│   ├── training/      Phase 2 baselines, Phase 3 neural model
│   └── eval/          Standalone evaluation against the LoRA's held-out sets
├── src/               Installable inference package
├── filter/            Open WebUI request-hook filter
├── data/              (gitignored) raw pulls and clean splits
└── models/            (gitignored) versioned run artifacts
```

## Related

- [openwebui-safety-filters](https://github.com/beaudamore/openwebui-safety-filters) — the live filters, including the inlet prompt-injection filter this complements
- [safety](https://github.com/beaudamore/safety) — the prompt-injection and content-safety LoRAs; the dataset recipe and eval sets here are taken from its notebooks
- [email-classifier-nn](https://github.com/beaudamore/email-classifier-nn) — the pipeline this repo mirrors
