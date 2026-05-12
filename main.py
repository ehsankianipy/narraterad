"""
main.py — NarrateRad
====================
Local FastAPI web server.

Endpoints:
  GET  /          — serves the frontend UI
  WS   /ws        — real-time audio → transcription pipeline
  POST /structure — structures transcript into MT-ready preliminary report

Run with:
  uv run uvicorn main:app --reload --port 8000
"""

import asyncio
import json
import traceback
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from nlp import check_all
from radlex import standardise
from structure import OllamaNotRunningError, PatientInfo, structure_report_stream
from transcribe import Transcriber

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 16_000
TRANSCRIPTION_INTERVAL: int = 10
OVERLAP_SECONDS: int = 2
FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="NarrateRad")
transcriber = Transcriber()


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/")
async def index() -> HTMLResponse:
    if not FRONTEND_PATH.exists():
        return HTMLResponse("<h2>Frontend not found.</h2>")
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@app.post("/structure")
async def structure(payload: dict) -> dict:
    text = payload.get("text", "").strip()
    if not text:
        return {"error": "No text provided"}

    patient = PatientInfo(
        name=payload.get("patient_name", ""),
        age=payload.get("patient_age", ""),
        mr_number=payload.get("mr_number", ""),
        referring_physician=payload.get("referring_physician", ""),
    )

    try:
        report = ""
        async for chunk in structure_report_stream(text, patient):
            report += chunk
        return {"structured": report}
    except OllamaNotRunningError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws")
async def websocket_transcribe(ws: WebSocket) -> None:
    await ws.accept()

    # Heartbeat — keeps connection alive during long Whisper transcriptions
    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(10)
            await _send(ws, {"type": "ping"})

    heartbeat_task = asyncio.create_task(heartbeat())

    buf: np.ndarray = np.zeros(0, dtype=np.float32)
    interval_samples = SAMPLE_RATE * TRANSCRIPTION_INTERVAL
    overlap_samples = SAMPLE_RATE * OVERLAP_SECONDS
    transcribing = False

    await _send(ws, {"type": "status", "message": "Connected — start speaking"})

    try:
        while True:
            raw = await ws.receive_bytes()
            incoming = np.frombuffer(raw, dtype=np.float32).copy()
            buf = np.concatenate([buf, incoming])

            if len(buf) >= interval_samples and not transcribing:
                audio_chunk = buf[:interval_samples].copy()
                buf = buf[interval_samples - overlap_samples:]

                transcribing = True
                await _send(ws, {"type": "status", "message": "Transcribing..."})

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, transcriber.transcribe, audio_chunk
                )
                transcribing = False

                if not result.is_empty():

                    # ── Transcript update ──────────────────────────────────
                    await _send(ws, {
                        "type": "transcript_update",
                        "words": [
                            {
                                "text": w.text,
                                "start": w.start,
                                "end": w.end,
                                "probability": w.probability,
                                "flagged": w.flagged,
                            }
                            for w in result.words
                        ],
                        "full_text": result.text,
                        "flag_rate": round(result.flag_rate, 3),
                        "language": result.language,
                    })

                    # ── NLP checks ─────────────────────────────────────────
                    nlp_flags = check_all(result.text)
                    if nlp_flags:
                        await _send(ws, {
                            "type": "nlp_flags",
                            "flags": [f.to_dict() for f in nlp_flags],
                        })

                    # ── RadLex standardisation ─────────────────────────────
                    standardised_text, radlex_corrections = standardise(result.text)
                    if radlex_corrections:
                        await _send(ws, {
                            "type": "radlex_corrections",
                            "corrections": [
                                {
                                    "original": c.original,
                                    "standardised": c.standardised,
                                    "concept": c.radlex_concept,
                                }
                                for c in radlex_corrections
                            ],
                            "standardised_text": standardised_text,
                        })

                await _send(ws, {"type": "status", "message": "Listening..."})

    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "status", "message": f"Error: {str(e)}"})
    finally:
        heartbeat_task.cancel()


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _send(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass
