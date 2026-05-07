"""
structure.py — NarrateRad
==========================
LLM structuring module — converts raw radiology dictation into a formatted report.

Calls Ollama (llama3.1) running locally on localhost:11434.
No data leaves the machine.

Functions:
    structure_report(text)         Synchronous — returns full report string
    structure_report_stream(text)  Async generator — yields chunks for streaming UI

Usage:
    from structure import structure_report
    report = structure_report("No pneumothorax. Mild left pleural effusion.")
    print(report)

Make sure Ollama is running before calling these:
    ollama serve &
    ollama pull llama3.1
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL: str = "http://localhost:11434/api/generate"
MODEL: str = "llama3.1"
TIMEOUT: float = 120.0  # seconds — large models can be slow on first token

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a radiology report assistant. Your task is to take raw dictation from a radiologist and format it into a clean, structured radiology report.

Output the report using exactly these four section headers, in this order:

CLINICAL INDICATION:
[The reason for the examination. Write "Not provided" if not mentioned in the dictation.]

TECHNIQUE:
[The imaging modality and technique. Write "Not specified" if not mentioned.]

FINDINGS:
[Organise findings by organ system (e.g. Lungs, Heart, Mediastinum, Bones, Soft Tissues). Use formal radiology language. List each system on a new line with its findings.]

IMPRESSION:
[A concise numbered summary of the key findings and their clinical significance. Maximum 5 points.]

Strict rules:
- Never invent findings not present in the dictation
- Never omit findings that are present in the dictation
- Preserve all laterality (left/right) exactly as dictated — if laterality is ambiguous, add [LATERALITY CHECK] after that finding
- Use formal radiology terminology throughout
- If the dictation is too incomplete to fill a section, write "Insufficient information"
- Do not add commentary, preamble, or explanation — output the structured report only"""
"[The reason for the examination. Write \"Not provided\" if not explicitly stated "
"in the dictation. Do NOT guess or infer the indication.]"


# ── Exceptions ────────────────────────────────────────────────────────────────


class OllamaNotRunningError(Exception):
    """Raised when Ollama is not reachable on localhost:11434."""

    def __str__(self) -> str:
        return (
            "Ollama is not running. Start it with: ollama serve\n"
            "Then make sure llama3.1 is pulled: ollama pull llama3.1"
        )


# ── Core functions ────────────────────────────────────────────────────────────


def structure_report(dictation: str) -> str:
    """
    Convert raw radiology dictation into a structured report.

    Calls Ollama synchronously — blocks until the full report is returned.
    Suitable for batch processing or testing. Use structure_report_stream()
    for the live UI to show text appearing word by word.

    Parameters
    ----------
    dictation : str
        Raw transcript text from the radiologist's dictation.

    Returns
    -------
    str
        Formatted radiology report with four sections.

    Raises
    ------
    OllamaNotRunningError
        If Ollama is not reachable on localhost:11434.
    """
    dictation = dictation.strip()
    if not dictation:
        return "No dictation provided."

    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": f"Structure this radiology dictation into a formal report:\n\n{dictation}",
        "stream": False,
    }

    try:
        response = httpx.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except httpx.ConnectError:
        raise OllamaNotRunningError()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama returned an error: {e.response.status_code}") from e


async def structure_report_stream(dictation: str) -> AsyncGenerator[str, None]:
    """
    Convert raw radiology dictation into a structured report, streaming the output.

    Yields text chunks as they arrive from Ollama so the UI can display
    the report appearing word by word rather than waiting for the full response.

    Parameters
    ----------
    dictation : str
        Raw transcript text from the radiologist's dictation.

    Yields
    ------
    str
        Text chunks from the LLM response.

    Raises
    ------
    OllamaNotRunningError
        If Ollama is not reachable on localhost:11434.
    """
    dictation = dictation.strip()
    if not dictation:
        yield "No dictation provided."
        return

    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": f"Structure this radiology dictation into a formal report:\n\n{dictation}",
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
        raise RuntimeError(f"Ollama returned an error: {e.response.status_code}") from e


# ── Smoke test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import asyncio

    TEST_DICTATION = (
        "PA chest x-ray. No pneumothorax. There is a moderate left pleural effusion. "
        "The right lung is clear. Heart size is upper limits of normal. "
        "The mediastinum is not widened. No bony abnormality identified. "
        "Impression: moderate left pleural effusion, otherwise unremarkable."
    )

    print("=" * 60)
    print("NarrateRad -- structure.py smoke test")
    print("=" * 60)
    print("\nDictation input:")
    print(f"  {TEST_DICTATION}")
    print("\nStructuring with Llama 3.1...\n")
    print("-" * 60)

    async def stream_test() -> None:
        try:
            async for chunk in structure_report_stream(TEST_DICTATION):
                print(chunk, end="", flush=True)
            print("\n" + "-" * 60)
            print("\nStreaming test passed.")
        except OllamaNotRunningError as e:
            print(f"\nError: {e}")

    asyncio.run(stream_test())
