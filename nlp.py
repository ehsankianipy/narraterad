"""
nlp.py — NarrateRad
====================
Rule-based clinical NLP — four checkers for common radiology dictation errors.

All functions accept a plain text string (the accumulated transcript) and return
a list of ClinicalFlag objects. No external ML models required — pure Python rules.

Functions:
    check_laterality(text)      Left/right conflicts for the same structure
    check_negation(text)        Finding mentioned both negated and affirmed
    check_contradictions(text)  Opposing descriptors (normal/abnormal, mild/severe)
    check_critical_findings(text) Emergency keywords requiring immediate attention
    check_all(text)             Runs all four and returns combined results

Usage:
    from nlp import check_all, ClinicalFlag
    flags = check_all(transcript_text)
    for f in flags:
        print(f.severity, f.message)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


# ── Data structure ────────────────────────────────────────────────────────────


@dataclass
class ClinicalFlag:
    """
    A single clinical warning raised by the NLP layer.

    Attributes
    ----------
    flag_type : str
        One of: 'laterality', 'negation', 'contradiction', 'critical'
    message : str
        Human-readable description shown in the UI.
    severity : 'warning' | 'critical'
        'critical' triggers the red alert banner; 'warning' shows inline.
    span_text : str
        The relevant text fragment that triggered this flag.
    """

    flag_type: str
    message: str
    severity: Literal["warning", "critical"]
    span_text: str

    def to_dict(self) -> dict:
        return {
            "flag_type": self.flag_type,
            "message": self.message,
            "severity": self.severity,
            "span_text": self.span_text,
        }


# ── Vocabulary constants ──────────────────────────────────────────────────────

# Anatomical structures that commonly have a left/right side in radiology
LATERALISED_STRUCTURES: list[str] = [
    "lung", "upper lobe", "lower lobe", "middle lobe",
    "pleura", "pleural effusion", "pleural space",
    "hemithorax", "pneumothorax",
    "hilum", "hilar",
    "kidney", "renal",
    "adrenal", "adrenal gland",
    "shoulder", "hip", "knee", "ankle", "wrist", "elbow",
    "ovary", "ovarian",
    "breast",
    "axilla", "axillary",
    "rib", "ribs",
    "consolidation",
    "opacity",
    "effusion",
    "lesion",
    "nodule",
    "mass",
    "lobe",
]

# Words/phrases that negate a finding
NEGATION_TERMS: list[str] = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bno evidence of\b",
    r"\bno sign of\b",
    r"\bnegative for\b",
    r"\bfree of\b",
]

# Findings tracked for negation inconsistency
TRACKED_FINDINGS: list[str] = [
    "pneumothorax",
    "pleural effusion",
    "effusion",
    "consolidation",
    "atelectasis",
    "opacity",
    "fracture",
    "dislocation",
    "mass",
    "nodule",
    "lesion",
    "lymphadenopathy",
    "cardiomegaly",
    "obstruction",
    "haemorrhage",
    "hemorrhage",
    "infarct",
    "oedema",
    "edema",
]

# Pairs of opposing terms that shouldn't coexist for the same structure
CONTRADICTORY_PAIRS: list[tuple[str, str]] = [
    ("normal", "abnormal"),
    ("unremarkable", "abnormal"),
    ("mild", "severe"),
    ("small", "large"),
    ("clear", "opacified"),
    ("intact", "fractured"),
    ("regular", "irregular"),
    ("homogeneous", "heterogeneous"),
    ("acute", "chronic"),        # flag — may be intentional (acute on chronic)
]

# Keywords that require immediate clinical escalation
CRITICAL_TERMS: list[str] = [
    "tension pneumothorax",
    "pneumothorax",
    "pulmonary embolism",
    "aortic dissection",
    "intracranial haemorrhage",
    "intracranial hemorrhage",
    "subarachnoid haemorrhage",
    "subarachnoid hemorrhage",
    "subdural haematoma",
    "subdural hematoma",
    "extradural haematoma",
    "extradural hematoma",
    "epidural haematoma",
    "epidural hematoma",
    "haemorrhage",
    "hemorrhage",
    "cauda equina",
    "cord compression",
    "bowel perforation",
    "free air",
    "pneumoperitoneum",
    "ischaemic stroke",
    "ischemic stroke",
    "infarct",
]

# Bilateral phrases — these mark intentional bilateral descriptions, not errors
BILATERAL_PHRASES: list[str] = [
    r"\bbilateral\b",
    r"\bbilaterally\b",
    r"\bboth\b",
    r"\bleft and right\b",
    r"\bright and left\b",
]


# ── Helper functions ──────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Lowercase and normalise whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _is_negated(text: str, finding: str) -> bool:
    """
    Returns True if the finding appears with a negation term
    within 40 characters before it.
    """
    pattern = (
        r"\b(?:no|not|without|absent|free of|no evidence of|no sign of)"
        r".{0,40}\b"
        + re.escape(finding)
        + r"\b"
    )
    return bool(re.search(pattern, text))


def _is_affirmed(text: str, finding: str) -> bool:
    """
    Returns True if the finding appears WITHOUT a negation term
    in the 50 characters before it.
    """
    for match in re.finditer(r"\b" + re.escape(finding) + r"\b", text):
        window = text[max(0, match.start() - 50) : match.start()]
        has_neg = any(re.search(neg, window) for neg in NEGATION_TERMS)
        if not has_neg:
            return True
    return False


def _near_bilateral(text: str, pos: int, window: int = 60) -> bool:
    """
    Returns True if a bilateral phrase appears within `window` characters
    of position `pos` — indicating an intentional bilateral description.
    """
    snippet = text[max(0, pos - window) : min(len(text), pos + window)]
    return any(re.search(phrase, snippet) for phrase in BILATERAL_PHRASES)


# ── Checkers ──────────────────────────────────────────────────────────────────


def check_laterality(text: str) -> list[ClinicalFlag]:
    """
    Detect when the same anatomical structure appears with BOTH left and
    right modifiers — suggesting the wrong side may have been dictated.

    Intentional bilateral descriptions ("bilateral kidneys", "left and right
    lungs") are excluded.
    """
    flags: list[ClinicalFlag] = []
    t = _normalise(text)

    for structure in LATERALISED_STRUCTURES:
        struct_re = r"\b" + re.escape(structure) + r"\b"

        # Find all "left <structure>" and "right <structure>" occurrences
        left_pattern = r"\bleft\b.{0,30}" + struct_re
        right_pattern = r"\bright\b.{0,30}" + struct_re

        left_matches = list(re.finditer(left_pattern, t))
        right_matches = list(re.finditer(right_pattern, t))

        if not (left_matches and right_matches):
            continue

        # Skip if any match is near a bilateral qualifier
        all_positions = [m.start() for m in left_matches + right_matches]
        if any(_near_bilateral(t, pos) for pos in all_positions):
            continue

        flags.append(
            ClinicalFlag(
                flag_type="laterality",
                message=(
                    f'Both "left {structure}" and "right {structure}" found — '
                    f"verify the correct side was dictated."
                ),
                severity="warning",
                span_text=f"left/right {structure}",
            )
        )

    return flags


def check_negation(text: str) -> list[ClinicalFlag]:
    """
    Detect findings that appear both negated and affirmed in the same
    transcript — e.g. "no pneumothorax" earlier, then "pneumothorax noted".
    """
    flags: list[ClinicalFlag] = []
    t = _normalise(text)

    for finding in TRACKED_FINDINGS:
        if not re.search(r"\b" + re.escape(finding) + r"\b", t):
            continue

        negated = _is_negated(t, finding)
        affirmed = _is_affirmed(t, finding)

        if negated and affirmed:
            flags.append(
                ClinicalFlag(
                    flag_type="negation",
                    message=(
                        f'"{finding}" appears both negated and affirmed — '
                        f"possible contradiction in findings."
                    ),
                    severity="warning",
                    span_text=finding,
                )
            )

    return flags


def check_contradictions(text: str) -> list[ClinicalFlag]:
    """
    Detect opposing descriptors appearing together — e.g. "normal" and
    "abnormal", "mild" and "severe" — which may indicate a dictation error.
    """
    flags: list[ClinicalFlag] = []
    t = _normalise(text)

    for term_a, term_b in CONTRADICTORY_PAIRS:
        has_a = bool(re.search(r"\b" + re.escape(term_a) + r"\b", t))
        has_b = bool(re.search(r"\b" + re.escape(term_b) + r"\b", t))

        if has_a and has_b:
            flags.append(
                ClinicalFlag(
                    flag_type="contradiction",
                    message=(
                        f'Opposing terms "{term_a}" and "{term_b}" both appear — '
                        f"verify this is intentional."
                    ),
                    severity="warning",
                    span_text=f"{term_a} / {term_b}",
                )
            )

    return flags


def check_critical_findings(text: str) -> list[ClinicalFlag]:
    """
    Scan for emergency keywords requiring immediate clinical escalation.
    Matches whole words only to avoid false positives from substrings.

    Note: critical findings inside negation ("no pneumothorax") are still
    flagged — the radiologist should explicitly confirm the negative finding.
    """
    flags: list[ClinicalFlag] = []
    t = _normalise(text)

    # Sort by length (longest first) so multi-word terms match before sub-terms
    sorted_terms = sorted(CRITICAL_TERMS, key=len, reverse=True)
    matched: set[str] = set()

    for term in sorted_terms:
        # Skip if a longer term containing this one already matched
        if any(term in m for m in matched):
            continue

        if term.count(" ") > 0:
            # Multi-word: substring match on normalised text
            if term in t:
                matched.add(term)
                flags.append(
                    ClinicalFlag(
                        flag_type="critical",
                        message=f'Critical finding: "{term}" — confirm and escalate if positive.',
                        severity="critical",
                        span_text=term,
                    )
                )
        else:
            # Single word: whole-word match only
            if re.search(r"\b" + re.escape(term) + r"\b", t):
                matched.add(term)
                flags.append(
                    ClinicalFlag(
                        flag_type="critical",
                        message=f'Critical finding: "{term}" — confirm and escalate if positive.',
                        severity="critical",
                        span_text=term,
                    )
                )

    return flags


def check_all(text: str) -> list[ClinicalFlag]:
    """
    Run all four checkers and return the combined flag list.
    This is the main entry point called by main.py after each transcription.

    Critical flags are placed first so they're shown at the top of the UI.
    """
    flags: list[ClinicalFlag] = []
    flags.extend(check_critical_findings(text))  # critical first
    flags.extend(check_laterality(text))
    flags.extend(check_negation(text))
    flags.extend(check_contradictions(text))
    return flags


# ── Smoke test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":

    TEST_CASES: list[tuple[str, str]] = [
        (
            "laterality",
            "There is a left pleural effusion. The right pleural effusion "
            "is small. No pneumothorax.",
        ),
        (
            "negation",
            "No pneumothorax identified. The heart is normal in size. "
            "There is a small pneumothorax at the apex.",
        ),
        (
            "contradiction",
            "The cardiac silhouette appears normal. There is mild cardiomegaly. "
            "Lungs are clear.",
        ),
        (
            "critical",
            "There is a moderate right-sided pneumothorax with mediastinal shift "
            "raising concern for tension pneumothorax.",
        ),
        (
            "clean",
            "No acute cardiopulmonary abnormality. Heart size is normal. "
            "Lungs are clear bilaterally. No pleural effusion.",
        ),
        (
            "bilateral ok",
            "Bilateral pleural effusions are present, left greater than right.",
        ),
    ]

    print("=" * 60)
    print("NarrateRad — nlp.py smoke test")
    print("=" * 60)

    for label, text in TEST_CASES:
        print(f"\n[{label.upper()}]")
        print(f"  Text: {text[:80]}...")
        flags = check_all(text)
        if not flags:
            print("  Result: No flags raised")
        else:
            for f in flags:
                icon = "🔴" if f.severity == "critical" else "🟡"
                print(f"  {icon} [{f.flag_type}] {f.message}")
