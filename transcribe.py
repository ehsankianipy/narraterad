"""
transcribe.py — NarrateRad (mlx-whisper edition)
=================================================
Whisper transcription with per-word confidence scores and hallucination filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mlx_whisper
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_REPO: str = "mlx-community/whisper-large-v3-mlx"
SAMPLE_RATE: int = 16_000
CONFIDENCE_THRESHOLD: float = 0.6
MIN_AUDIO_SECONDS: float = 1.0

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

    # Known Whisper silence hallucinations — matched after stripping punctuation
    HALLUCINATION_PHRASES = [
        "thank you",
        "thanks for watching",
        "subtitles by",
        "subscribe",
        "like and subscribe",
        "see you next time",
        "please subscribe",
        "transcribed by",
        "hello",
        "bye",
        "goodbye",
        "you",
    ]

    def __init__(
        self,
        model_repo: str = MODEL_REPO,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        use_radiology_prompt: bool = True,
    ) -> None:
        self._model_repo = model_repo
        self._confidence_threshold = confidence_threshold
        self._initial_prompt = RADIOLOGY_PROMPT if use_radiology_prompt else None

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_AUDIO_SECONDS:
            return TranscriptionResult(words=[], language="en")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_repo,
            word_timestamps=True,
            initial_prompt=self._initial_prompt,
            language="en",
            verbose=False,
            condition_on_previous_text=False,
        )

        words = self._extract_words(result)
        language = result.get("language", "en")
        return TranscriptionResult(words=words, language=language)

    def _extract_words(self, result: dict) -> list[Word]:
        words: list[Word] = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                text = w.get("word", "").strip()
                if not text:
                    continue
                prob = float(w.get("probability", 1.0))
                words.append(
                    Word(
                        text=text,
                        start=round(float(w.get("start", 0.0)), 3),
                        end=round(float(w.get("end", 0.0)), 3),
                        probability=round(prob, 4),
                        flagged=prob < self._confidence_threshold,
                    )
                )
        return self._filter_hallucinations(words)

    @staticmethod
    def _clean(s: str) -> str:
        """Strip punctuation and lowercase for comparison."""
        return re.sub(r"[^\w\s]", "", s.lower()).strip()

    def _filter_hallucinations(self, words: list[Word]) -> list[Word]:
        """
        Remove known Whisper hallucination patterns:
        1. Known filler phrases (matched after stripping punctuation)
        2. Consecutive duplicate words
        3. Repeated sentences
        """
        if not words:
            return words

        # ── Step 1: remove known filler phrases ───────────────────────────
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

        # ── Step 2: remove consecutive duplicate words ────────────────────
        deduped: list[Word] = []
        for w in words:
            if not deduped or self._clean(w.text) != self._clean(deduped[-1].text):
                deduped.append(w)
        words = deduped

        # ── Step 3: remove repeated sentences ────────────────────────────
        # Split into sentences on sentence-ending punctuation
        sentences: list[list[Word]] = []
        current: list[Word] = []
        for w in words:
            current.append(w)
            if re.search(r"[.?!]$", w.text.strip()):
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)

        # Keep only non-duplicate consecutive sentences
        unique: list[list[Word]] = []
        for sent in sentences:
            sent_text = self._clean(" ".join(w.text for w in sent))
            if not unique or sent_text != self._clean(" ".join(w.text for w in unique[-1])):
                unique.append(sent)

        # Flatten back to words
        return [w for sent in unique for w in sent]


# ── Smoke test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sounddevice as sd

    RECORD_SECONDS = 8

    print("=" * 60)
    print("NarrateRad -- transcribe.py smoke test")
    print("=" * 60)
    print(f"\nRecording {RECORD_SECONDS} seconds -- speak a radiology finding.\n")

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