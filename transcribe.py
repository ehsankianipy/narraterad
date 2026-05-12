"""
transcribe.py — NarrateRad
==========================
Cross-platform Whisper transcription module.

Automatically selects the right backend:
  - Apple Silicon Mac  → mlx-whisper  (Metal GPU, fast)
  - Windows / Intel    → faster-whisper (CPU, int8)

The rest of the app (main.py, nlp.py, structure.py) is identical on both
platforms — only this file differs in its backend.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass

import numpy as np
from radlex import RADLEX_PROMPT
RADIOLOGY_PROMPT = RADLEX_PROMPT

# ── Platform detection ────────────────────────────────────────────────────────

IS_APPLE_SILICON: bool = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 16_000
CONFIDENCE_THRESHOLD: float = 0.6
MIN_AUDIO_SECONDS: float = 1.0

# Apple Silicon model
MLX_MODEL_REPO: str = "mlx-community/whisper-large-v3-mlx"

# Windows / CPU model — medium gives the best speed/accuracy balance on CPU
FW_MODEL_SIZE: str = "medium"

RADIOLOGY_PROMPT: str = (
    "Radiology report dictation. "
    "Medical terms: pneumothorax, effusion, consolidation, atelectasis, "
    "cardiomegaly, mediastinum, hilum, pleural, pericardial, hepatic, "
    "splenic, renal, aortic, pulmonary embolism, haemorrhage, infarct, "
    "comminuted, displaced, cortex, diaphragm, costophrenic, trachea, "
    "lytic, sclerotic, lucency, opacity, infiltrate, nodule, mass."
)


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float
    flagged: bool

    def __str__(self) -> str:
        return f"{self.text}[?]" if self.flagged else self.text


@dataclass
class TranscriptionResult:
    words: list[Word]
    language: str

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words if w.text)

    @property
    def clean_text(self) -> str:
        parts = []
        for w in self.words:
            if not w.text:
                continue
            parts.append(f"[?{w.text}?]" if w.flagged else w.text)
        return " ".join(parts)

    @property
    def flagged_words(self) -> list[Word]:
        return [w for w in self.words if w.flagged]

    @property
    def flag_rate(self) -> float:
        if not self.words:
            return 0.0
        return len(self.flagged_words) / len(self.words)

    def is_empty(self) -> bool:
        return len(self.words) == 0


# ── Transcriber ───────────────────────────────────────────────────────────────


class Transcriber:
    """
    Cross-platform Whisper transcriber.

    On Apple Silicon: uses mlx-whisper (Metal GPU acceleration).
    On Windows/CPU:   uses faster-whisper with int8 quantisation.

    Both backends return identical TranscriptionResult objects so the
    rest of the app doesn't need to know which backend is running.
    """

    HALLUCINATION_PHRASES = [
        "thank you",
        "thanks for watching",
        "subtitles by",
        "subscribe",
        "like and subscribe",
        "see you next time",
        "please subscribe",
        "for watching",
        "transcribed by",
        "hello",
        "bye",
        "goodbye",
    ]

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        use_radiology_prompt: bool = True,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._initial_prompt = RADIOLOGY_PROMPT if use_radiology_prompt else None
        self._fw_model = None  # lazy-loaded for faster-whisper

        backend = "mlx-whisper (Apple Silicon)" if IS_APPLE_SILICON else "faster-whisper (CPU)"
        print(f"[transcribe] Backend: {backend}")

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe a mono float32 array at 16kHz.
        Automatically uses the right backend for this platform.
        """
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_AUDIO_SECONDS:
            return TranscriptionResult(words=[], language="en")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if IS_APPLE_SILICON:
            return self._transcribe_mlx(audio)
        else:
            return self._transcribe_fw(audio)

    # ── Apple Silicon backend ─────────────────────────────────────────────────

    def _transcribe_mlx(self, audio: np.ndarray) -> TranscriptionResult:
        import mlx_whisper  # type: ignore

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=MLX_MODEL_REPO,
            word_timestamps=True,
            initial_prompt=self._initial_prompt,
            language="en",
            verbose=False,
            condition_on_previous_text=False,
        )

        words = self._extract_words_mlx(result)
        return TranscriptionResult(
            words=self._filter_hallucinations(words),
            language=result.get("language", "en"),
        )

    def _extract_words_mlx(self, result: dict) -> list[Word]:
        words: list[Word] = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                text = w.get("word", "").strip()
                if not text:
                    continue
                prob = float(w.get("probability", 1.0))
                words.append(Word(
                    text=text,
                    start=round(float(w.get("start", 0.0)), 3),
                    end=round(float(w.get("end", 0.0)), 3),
                    probability=round(prob, 4),
                    flagged=prob < self._confidence_threshold,
                ))
        return words

    # ── Windows / CPU backend ─────────────────────────────────────────────────

    def _transcribe_fw(self, audio: np.ndarray) -> TranscriptionResult:
        if self._fw_model is None:
            print(f"[transcribe] Loading faster-whisper {FW_MODEL_SIZE}...")
            from faster_whisper import WhisperModel  # type: ignore
            self._fw_model = WhisperModel(
                FW_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
            print("[transcribe] Model ready.")

        segments, info = self._fw_model.transcribe(
            audio,
            word_timestamps=True,
            initial_prompt=self._initial_prompt,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,  # faster-whisper has built-in VAD
        )

        words = self._extract_words_fw(segments)
        return TranscriptionResult(
            words=self._filter_hallucinations(words),
            language=info.language,
        )

    def _extract_words_fw(self, segments: object) -> list[Word]:
        words: list[Word] = []
        for segment in segments:  # type: ignore
            if not segment.words:
                continue
            for w in segment.words:
                text = w.word.strip()
                if not text:
                    continue
                prob = float(w.probability)
                words.append(Word(
                    text=text,
                    start=round(float(w.start), 3),
                    end=round(float(w.end), 3),
                    probability=round(prob, 4),
                    flagged=prob < self._confidence_threshold,
                ))
        return words

    # ── Hallucination filter ──────────────────────────────────────────────────

    @staticmethod
    def _clean(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s.lower()).strip()

    def _filter_hallucinations(self, words: list[Word]) -> list[Word]:
        if not words:
            return words

        for phrase in self.HALLUCINATION_PHRASES:
            phrase_words = phrase.split()
            filtered: list[Word] = []
            i = 0
            while i < len(words):
                window = [
                    self._clean(words[j].text)
                    for j in range(i, min(i + len(phrase_words), len(words)))
                ]
                if window == phrase_words:
                    i += len(phrase_words)
                else:
                    filtered.append(words[i])
                    i += 1
            words = filtered

        deduped: list[Word] = []
        for w in words:
            if not deduped or self._clean(w.text) != self._clean(deduped[-1].text):
                deduped.append(w)
        words = deduped

        sentences: list[list[Word]] = []
        current: list[Word] = []
        for w in words:
            current.append(w)
            if re.search(r"[.?!]$", w.text.strip()):
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)

        unique: list[list[Word]] = []
        for sent in sentences:
            sent_text = self._clean(" ".join(w.text for w in sent))
            if not unique or sent_text != self._clean(" ".join(w.text for w in unique[-1])):
                unique.append(sent)

        return [w for sent in unique for w in sent]


# ── Smoke test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sounddevice as sd

    RECORD_SECONDS = 8
    backend = "mlx-whisper" if IS_APPLE_SILICON else "faster-whisper"

    print("=" * 60)
    print(f"NarrateRad -- transcribe.py smoke test ({backend})")
    print("=" * 60)
    print(f"\nRecording {RECORD_SECONDS} seconds -- dictate a finding.\n")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
    )
    sd.wait()

    print("Transcribing...\n")
    t = Transcriber()
    result = t.transcribe(audio[:, 0])

    if result.is_empty():
        print("No words detected.")
    else:
        print(f"Language : {result.language}")
        print(f"Words    : {len(result.words)}")
        print(f"Flagged  : {len(result.flagged_words)} ({result.flag_rate:.1%})")
        print()
        for w in result.words:
            flag = "  <- FLAGGED" if w.flagged else ""
            print(f"  {w.start:5.2f}s  {w.text:<25} p={w.probability:.3f}{flag}")
        print(f"\nText: {result.text}")
