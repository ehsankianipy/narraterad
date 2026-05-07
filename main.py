"""
main.py — NarrateRad
====================
Local FastAPI web server.

Endpoints:
  GET  /          — serves the frontend UI
  WS   /ws        — real-time audio → transcription pipeline

WebSocket protocol:
  Client → Server : Binary frames (float32 PCM at 16kHz, raw ArrayBuffer)
  Server → Client : JSON messages —
    {type: "status",           message: str}
    {type: "transcript_update", words: [...], full_text: str, flag_rate: float}

Run with:
  uv run uvicorn main:app --reload --port 8000
Then open: http://localhost:8000
"""

from structure import structure_report_stream, OllamaNotRunningError
import traceback
from nlp import check_all
import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from transcribe import Transcriber

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 16_000

# How many seconds of audio to accumulate before transcribing.
# 10s gives a good balance of responsiveness and accuracy.
TRANSCRIPTION_INTERVAL: int = 15

# Seconds of audio kept from previous chunk to prevent word cutoffs.
OVERLAP_SECONDS: int = 2

FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="NarrateRad")

# Single shared transcriber — model loads on first transcription call.
# Subsequent calls reuse the loaded model (fast).
transcriber = Transcriber()


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/")
async def index() -> HTMLResponse:
    """Serve the frontend UI."""
    if not FRONTEND_PATH.exists():
        return HTMLResponse(
            "<h2>Frontend not found.</h2>"
            "<p>Create <code>frontend/index.html</code> and restart the server.</p>"
        )
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))

@app.post("/structure")
async def structure(payload: dict) -> dict:
    text = payload.get("text", "").strip()
    if not text:
        return {"error": "No text provided"}
    try:
        report = ""
        async for chunk in structure_report_stream(text):
            report += chunk
        return {"structured": report}
    except OllamaNotRunningError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_transcribe(ws: WebSocket) -> None:
    """
    Real-time transcription WebSocket.

    Receives raw float32 PCM audio chunks from the browser, accumulates them
    into 10-second windows, transcribes with mlx-whisper, and streams word-level
    results back to the client.
    """
    await ws.accept()

    # Audio accumulation buffer
    buf: np.ndarray = np.zeros(0, dtype=np.float32)
    interval_samples = SAMPLE_RATE * TRANSCRIPTION_INTERVAL
    overlap_samples = SAMPLE_RATE * OVERLAP_SECONDS

    # Simple flag to avoid overlapping transcription calls
    transcribing = False

    await _send(ws, {"type": "status", "message": "Connected — start speaking"})

    try:
        while True:
            raw = await ws.receive_bytes()

            # Convert incoming bytes to float32 numpy array
            incoming = np.frombuffer(raw, dtype=np.float32).copy()
            buf = np.concatenate([buf, incoming])

            # Transcribe once we have enough audio and aren't already transcribing
            if len(buf) >= interval_samples and not transcribing:
                audio_chunk = buf[:interval_samples].copy()

                # Keep the overlap tail for next chunk
                buf = buf[interval_samples - overlap_samples :]

                transcribing = True
                await _send(ws, {"type": "status", "message": "Transcribing..."})

                # Run blocking transcription in a thread pool so we don't block
                # the async event loop (mlx_whisper.transcribe is synchronous)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, transcriber.transcribe, audio_chunk
                )

                transcribing = False

                if not result.is_empty():
                    await _send(
                        ws,
                        {
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
                        },
                    )

                    # ── NLP checks ──────────────────────────────────────
                    nlp_flags = check_all(result.text)
                    if nlp_flags:
                        await _send(ws, {
                            "type": "nlp_flags",
                            "flags": [f.to_dict() for f in nlp_flags],
                        })
                await _send(ws, {"type": "status", "message": "Listening..."})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
        await _send(ws, {"type": "status", "message": f"Error: {str(e)}"})


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _send(ws: WebSocket, payload: dict) -> None:
    """Send a JSON payload — swallows errors from closed connections."""
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass
