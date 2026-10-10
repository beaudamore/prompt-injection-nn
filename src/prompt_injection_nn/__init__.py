"""prompt_injection_nn: classify a chunk of text as SAFE or INJECTION with a reason.

Public API:

    from prompt_injection_nn import load_detector
    detector = load_detector()                 # newest run under models/, backend from gate.json
    verdict = detector.classify("...")         # Verdict(label, probability, reason, ...)
    verdict.contract                           # "SAFE" or "INJECTION: <reason>"
    detector.classify_batch([...])             # order-preserving

Output contract shared with the prompt-injection LoRA and the Open WebUI filters:
"SAFE" or "INJECTION: <reason>", reason one of REASONS.
"""

from .contract import REASONS, Verdict, contract_string
from .detector import Detector, load_detector
from .tokenizer import Tokenizer

__all__ = ["REASONS", "Verdict", "contract_string", "Detector", "load_detector", "Tokenizer"]
__version__ = "0.1.0"
