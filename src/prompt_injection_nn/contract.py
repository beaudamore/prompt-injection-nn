"""The output contract. Fixed; the LoRA and the Open WebUI filters depend on it."""

from dataclasses import dataclass, field
from typing import Dict, Optional

REASONS = (
    "Override Attempt",
    "Fake System Injection",
    "Meta-Command Injection",
    "Jailbreak Technique",
    "Delimiter Confusion",
    "Privilege Escalation",
    "Embedded Instruction",
    "Role-Play Bypass",
    "Chain-of-Instructions",
)

SAFE = "SAFE"
INJECTION = "INJECTION"


def contract_string(label: str, reason: Optional[str]) -> str:
    """'SAFE' or 'INJECTION: <reason>'. Raises on a reason outside the taxonomy."""
    if label == SAFE:
        return SAFE
    if label != INJECTION:
        raise ValueError(f"unknown label {label!r}")
    if reason not in REASONS:
        raise ValueError(f"reason {reason!r} is not in the taxonomy")
    return f"{INJECTION}: {reason}"


@dataclass(frozen=True)
class Verdict:
    label: str                       # "SAFE" or "INJECTION"
    probability: float               # calibrated injection probability, 0..1
    threshold: float                 # the threshold that produced `label`
    budget: str                      # "f1" or "fpr_0.01" / "fpr_0.005" / "fpr_0.001"
    reason: Optional[str] = None     # one of REASONS when label is INJECTION, else None
    reason_scores: Dict[str, float] = field(default_factory=dict)   # softmax over REASONS when the backend has a reason head
    backend: str = ""                # "cnn_bigru", "tfidf_logreg" or "ensemble_mean"
    windows: int = 1                 # token windows scored for long inputs (max taken)
    components: Dict[str, float] = field(default_factory=dict)   # each backend's own probability, e.g. {"cnn_bigru": .82, "tfidf_logreg": .40}

    @property
    def contract(self) -> str:
        return contract_string(self.label, self.reason)

    @property
    def is_injection(self) -> bool:
        return self.label == INJECTION
