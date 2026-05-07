"""
capture.py — NarrateRad
=======================
Continuous microphone recording module.

Produces fixed-length float32 audio chunks at 16kHz (Whisper's required format)
with configurable overlap so words at chunk boundaries are never lost.

Includes simple energy-based VAD: chunks whose RMS falls below the threshold
(i.e. silence) are discarded before being passed to the callback.

Usage:
    stream = CaptureStream(on_chunk=my_callback)
    stream.start()
    ...
    stream.stop()
"""

import queue
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

# ── Defaults ──────────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 16_000        # Hz — Whisper requires 16kHz
CHUNK_SECONDS: int = 30          # Production default; shorter = faster feedback
OVERLAP_SECONDS: int = 2         # Overlap prevents word cutoffs at boundaries
VAD_THRESHOLD: float = 0.01      # RMS below this = silence, chunk is skipped
BLOCK_SIZE: int = 1_024          # Samples per sounddevice callback (latency vs overhead)


# ── Data structures ───────────────────────────────────────────────────────────


class CaptureStream:
    """
    Wraps a sounddevice InputStream into a simple chunk-based API.

    The caller supplies a callback that receives numpy arrays:
        on_chunk(audio: np.ndarray)  # shape (SAMPLE_RATE * chunk_seconds,), dtype float32

    Internally:
    - An audio callback pushes raw blocks onto a queue.
    - A background processor thread drains the queue, assembles the buffer,
      fires the callback when a full chunk is ready, then retains the overlap
      tail for the next chunk.

    Parameters
    ----------
    on_chunk : callable
        Called with each valid audio chunk (numpy float32 array, 16kHz).
    sample_rate : int
        Sample rate in Hz. Must match Whisper's expected 16kHz.
    chunk_seconds : int
        Duration of each audio chunk. 30s gives best Whisper accuracy;
        reduce to 5-10s for faster real-time feedback at the cost of some accuracy.
    overlap_seconds : int
        Seconds of audio retained from the previous chunk to prevent word cutoffs.
    vad_threshold : float
        RMS threshold. Chunks quieter than this are treated as silence and skipped.
    """

    def __init__(
        self,
        on_chunk: Callable[[np.ndarray], None],
        sample_rate: int = SAMPLE_RATE,
        chunk_seconds: int = CHUNK_SECONDS,
        overlap_seconds: int = OVERLAP_SECONDS,
        vad_threshold: float = VAD_THRESHOLD,
    ) -> None:
        self._on_chunk = on_chunk
        self._sample_rate = sample_rate
        self._chunk_samples = sample_rate * chunk_seconds
        self._overlap_samples = sample_rate * overlap_seconds
        self._vad_threshold = vad_threshold

        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._processor_thread: threading.Thread | None = None
        self._running: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone stream and start the processor thread."""
        if self._running:
            print("[capture] Already running — call stop() first.")
            return

        self._running = True
        self._buffer = np.zeros(0, dtype=np.float32)

        self._processor_thread = threading.Thread(
            target=self._process_audio,
            daemon=True,
            name="narraterad-capture-processor",
        )
        self._processor_thread.start()

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=BLOCK_SIZE,
        )
        self._stream.start()

        print(
            f"[capture] Started — {self._sample_rate}Hz, "
            f"{self._chunk_samples // self._sample_rate}s chunks, "
            f"{self._overlap_samples // self._sample_rate}s overlap, "
            f"VAD threshold={self._vad_threshold}"
        )

    def stop(self) -> None:
        """Stop recording and clean up resources."""
        if not self._running:
            return

        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._processor_thread is not None:
            self._processor_thread.join(timeout=3.0)
            self._processor_thread = None

        print("[capture] Stopped.")

    # ── Internal methods ──────────────────────────────────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """
        sounddevice audio callback — called on a high-priority audio thread.
        Keep this fast: just copy and enqueue, nothing else.
        """
        if status:
            print(f"[capture] sounddevice status: {status}")
        # indata shape is (frames, channels) — flatten to mono
        self._audio_queue.put(indata[:, 0].copy())

    def _process_audio(self) -> None:
        """
        Processor thread: drains the queue, assembles chunks, applies VAD,
        fires the callback.
        """
        while self._running:
            try:
                block = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._buffer = np.concatenate([self._buffer, block])

            # Emit complete chunks as they fill up
            while len(self._buffer) >= self._chunk_samples:
                chunk = self._buffer[: self._chunk_samples].copy()

                # VAD check — skip silence
                rms = float(np.sqrt(np.mean(chunk**2)))
                if rms >= self._vad_threshold:
                    self._on_chunk(chunk)
                else:
                    print(f"[capture] Chunk skipped — silence (RMS={rms:.4f})")

                # Advance buffer, keep overlap tail for next chunk
                advance = self._chunk_samples - self._overlap_samples
                self._buffer = self._buffer[advance:]

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "CaptureStream":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


# ── Smoke test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("NarrateRad — capture.py smoke test")
    print("=" * 60)
    print()
    print("Recording for 35 seconds to capture one full 30s chunk.")
    print("Speak into your mic — silence will be skipped by VAD.")
    print("Press Ctrl+C to stop early.")
    print()

    chunks_received: list[dict] = []

    def on_chunk(audio: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(audio**2)))
        info = {
            "index": len(chunks_received) + 1,
            "samples": len(audio),
            "duration_s": len(audio) / SAMPLE_RATE,
            "rms": rms,
        }
        chunks_received.append(info)
        print(
            f"[test] ✓ Chunk {info['index']} received — "
            f"{info['duration_s']:.1f}s, "
            f"{info['samples']:,} samples, "
            f"RMS={info['rms']:.4f}"
        )

    stream = CaptureStream(on_chunk=on_chunk, chunk_seconds=30)

    try:
        stream.start()
        time.sleep(35)
    except KeyboardInterrupt:
        print("\n[test] Interrupted by user.")
    finally:
        stream.stop()

    print()
    print(f"Test complete — {len(chunks_received)} chunk(s) received.")
    if not chunks_received:
        print(
            "No chunks received. Either the recording was all silence "
            "(raise VAD_THRESHOLD) or it was interrupted before 30s."
        )
    else:
        for c in chunks_received:
            print(f"  Chunk {c['index']}: {c['duration_s']:.1f}s, RMS={c['rms']:.4f}")
