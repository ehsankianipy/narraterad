"""
structure.py — NarrateRad
==========================
LLM structuring module — converts raw radiology dictation into a
preliminary report formatted for Medical Transcriptionist (MT) handover.

Includes patient demographics, procedure heading, and IR-compatible structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncGenerator

import httpx

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL: str = "http://localhost:11434/api/generate"
MODEL: str = "llama3.1"
TIMEOUT: float = 120.0


# ── Patient info ──────────────────────────────────────────────────────────────


@dataclass
class PatientInfo:
    name: str = ""
    age: str = ""
    mr_number: str = ""
    referring_physician: str = ""

    def header(self) -> str:
        """Renders the patient demographics block for the report header."""
        lines = ["PRELIMINARY DICTATION", "─" * 40]
        if self.name:
            lines.append(f"Patient Name:        {self.name}")
        if self.age:
            lines.append(f"Age:                 {self.age}")
        if self.mr_number:
            lines.append(f"MR Number:           {self.mr_number}")
        if self.referring_physician:
            lines.append(f"Referring Physician: {self.referring_physician}")
        lines.append("─" * 40)
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return not any([self.name, self.age, self.mr_number, self.referring_physician])


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a radiology report assistant preparing a preliminary dictation for a Medical Transcriptionist (MT). Your task is to structure raw dictation into a clear, well-organised preliminary report that the MT can use to fill in the hospital reporting system.

Output the report using exactly these section headers in this order:

PROCEDURE:
[The imaging study or interventional procedure performed. For diagnostic imaging: modality, body part, and technique. For IR procedures: procedure name, access site, materials used (contrast, devices), and any immediate complications.]

CLINICAL INDICATION:
[The reason for the study. Write "Not provided" if not mentioned.]

FINDINGS:
[Organise findings by organ system. Each system on a new line. Use clear, formal radiology language. For IR procedures, include: pre-procedure status, intra-procedure findings, post-procedure status, and any complications.]

IMPRESSION:
[Concise numbered summary of key findings and clinical significance. Maximum 5 points. For IR: include technical success, clinical outcome, and follow-up recommendation.]

─────────────────────────────────
For MT: Please transcribe above findings into the reporting system exactly as dictated. Flag any unclear terms.
─────────────────────────────────

Strict rules:
- CRITICAL: Never invent findings not present in the dictation. Never add bilateral or any laterality unless explicitly stated by the radiologist. If only one side is mentioned, only write that side.
- If the radiologist says right only, write right only. Never assume the other side is involved.
- Never omit findings that are present
- Preserve all laterality exactly as dictated — if ambiguous, add [LATERALITY CHECK]
- Use formal radiology terminology
- If a section cannot be filled from the dictation, write "Refer to dictation"
- Output the structured report only — no commentary or preamble"""


# ── Exceptions ────────────────────────────────────────────────────────────────


class OllamaNotRunningError(Exception):
    def __str__(self) -> str:
        return (
            "Ollama is not running. Start it with: ollama serve\n"
            "Then make sure llama3.1 is pulled: ollama pull llama3.1"
        )


# ── Core functions ────────────────────────────────────────────────────────────


def _build_prompt(dictation: str, patient: PatientInfo | None) -> str:
    """Combine patient header and dictation into the LLM prompt."""
    parts = []
    if patient and not patient.is_empty():
        parts.append(patient.header())
        parts.append("")
    parts.append("Structure this dictation into a preliminary report for MT handover:")
    parts.append("")
    parts.append(dictation)
    return "\n".join(parts)


def structure_report(dictation: str, patient: PatientInfo | None = None) -> str:
    """
    Convert raw dictation into an MT-ready preliminary report.
    Synchronous — blocks until full report is returned.
    """
    dictation = dictation.strip()
    if not dictation:
        return "No dictation provided."

    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": _build_prompt(dictation, patient),
        "stream": False,
    }

    try:
        response = httpx.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json().get("response", "").strip()

        # Prepend patient header to the output if provided
        if patient and not patient.is_empty():
            return patient.header() + "\n\n" + result
        return result

    except httpx.ConnectError:
        raise OllamaNotRunningError()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama error: {e.response.status_code}") from e


async def structure_report_stream(
    dictation: str, patient: PatientInfo | None = None
) -> AsyncGenerator[str, None]:
    """
    Convert raw dictation into a preliminary report, streaming output.
    Yields text chunks for real-time display in the UI.
    """
    dictation = dictation.strip()
    if not dictation:
        yield "No dictation provided."
        return

    # Stream the patient header immediately before Ollama starts
    if patient and not patient.is_empty():
        yield patient.header() + "\n\n"

    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": _build_prompt(dictation, patient),
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    except httpx.ConnectError:
        raise OllamaNotRunningError()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama error: {e.response.status_code}") from e


# ── Smoke test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import asyncio

    patient = PatientInfo(
        name="Ahmed Khan",
        age="52 years",
        mr_number="MR-2024-00451",
        referring_physician="Dr. Sarah Ahmed",
    )

    dictation = (
        "CT chest with contrast. Moderate right pleural effusion. "
        "No pneumothorax. Heart size is normal. Mediastinum is central. "
        "No significant lymphadenopathy. Liver and spleen appear normal "
        "on the limited views. Impression: moderate right pleural effusion, "
        "recommend follow-up."
    )

    print("=" * 60)
    print("NarrateRad -- structure.py smoke test")
    print("=" * 60)

    async def run() -> None:
        try:
            async for chunk in structure_report_stream(dictation, patient):
                print(chunk, end="", flush=True)
            print("\n")
        except OllamaNotRunningError as e:
            print(f"\nError: {e}")

    asyncio.run(run())
