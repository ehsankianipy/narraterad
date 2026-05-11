"""
transcribe.py — NarrateRad (mlx-whisper edition)
=================================================
Whisper transcription module using mlx-whisper — Apple Silicon native.

mlx-whisper uses Apple's MLX framework and Metal GPU, making it significantly
faster than CPU-based alternatives on M-series chips.

Takes float32 numpy audio arrays (16kHz, mono) from capture.py and returns
a list of Word objects with text, timestamps, and per-word probability scores.

The noise scrubber is built in: words with probability below CONFIDENCE_THRESHOLD
are marked flagged=True and shown in amber in the UI for radiologist review.

Usage:
    t = Transcriber()
    result = t.transcribe(audio_array)
    print(result.text)
    print(result.flagged_words)
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx_whisper
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# Apple Silicon optimised model from mlx-community
MODEL_REPO: str = "mlx-community/whisper-large-v3-mlx"

SAMPLE_RATE: int = 16_000

# Words below this probability are flagged for radiologist review.
# 0.6 is conservative — catches genuinely ambiguous transcriptions without
# over-flagging normal speech. Tune based on your mic and environment.
CONFIDENCE_THRESHOLD: float = 0.6

# Radiology vocabulary prompt — primes Whisper to expect medical terms,
# significantly reduces misrecognitions of clinical language.
RADIOLOGY_PROMPT: str = (
    "Radiology report dictation. "
    "Medical terms: pneumothorax, effusion, consolidation, atelectasis, "
    "cardiomegaly, mediastinum, hilum, pleural, pericardial, hepatic, "
    "splenic, renal, aortic, pulmonary embolism, haemorrhage, infarct, "
    "comminuted, displaced, cortex, diaphragm, costophrenic, trachea, "
    "lytic, sclerotic, lucency, opacity, infiltrate, nodule, mass."
)

MIN_AUDIO_SECONDS: float = 1.0


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class Word:
    """
    A single transcribed word with timing and confidence information.

    Attributes
    ----------
    text : str
        The transcribed word (stripped of whitespace).
    start : float
        Start time in seconds within the audio chunk.
    end : float
        End time in seconds within the audio chunk.
    probability : float
        Per-word confidence from Whisper, 0.0 to 1.0.
    flagged : bool
        True if probability < CONFIDENCE_THRESHOLD. Shown in amber in the UI.
    """

    text: str
    start: float
    end: float
    probability: float
    flagged: bool

    def __str__(self) -> str:
        marker = " [?]" if self.flagged else ""
        return f"{self.text}{marker}"


@dataclass
class TranscriptionResult:
    """
    Full result from a single transcription call.

    Attributes
    ----------
    words : list[Word]
        All transcribed words, including flagged ones.
    language : str
        Auto-detected language code (e.g. 'en').
    """

    words: list[Word]
    language: str

    @property
    def text(self) -> str:
        """Full transcript as a plain string."""
        return " ".join(w.text for w in self.words if w.text)

    @property
    def clean_text(self) -> str:
        """Transcript with flagged words wrapped in [?...?] markers."""
        parts = []
        for w in self.words:
            if not w.text:
                continue
            parts.append(f"[?{w.text}?]" if w.flagged else w.text)
        return " ".join(parts)

    @property
    def flagged_words(self) -> list[Word]:
        """Words that need radiologist review."""
        return [w for w in self.words if w.flagged]

    @property
    def flag_rate(self) -> float:
        """Fraction of words flagged. High values suggest audio quality issues."""
        if not self.words:
            return 0.0
        return len(self.flagged_words) / len(self.words)

    def is_empty(self) -> bool:
        return len(self.words) == 0


# ── Transcriber ───────────────────────────────────────────────────────────────


class Transcriber:
    """
    Wraps mlx-whisper with noise scrubbing and a radiology context prompt.

    mlx-whisper loads the model lazily on the first transcribe() call.
    The model is cached in ~/.cache/huggingface/hub/ after first download.

    Parameters
    ----------
    model_repo : str
        Hugging Face repo for the mlx model. Defaults to large-v3.
        Use 'mlx-community/whisper-small-mlx' for faster, lower-accuracy testing.
    confidence_threshold : float
        Words below this probability are flagged for review.
    use_radiology_prompt : bool
        If True, primes Whisper with radiology vocabulary.
    """

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
        """
        Transcribe a mono float32 audio array sampled at 16kHz.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 array at 16kHz — the format produced by capture.py.

        Returns
        -------
        TranscriptionResult
            Words with timestamps, probabilities, and noise flags.
            Returns an empty result for silent or very short audio.
        """
        # Guard: too-short audio causes hallucinations
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_AUDIO_SECONDS:
            print(f"[transcribe] Audio too short ({duration:.2f}s) — skipping.")
            return TranscriptionResult(words=[], language="en")

        # Ensure correct dtype
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_repo,
            word_timestamps=True,
            initial_prompt=self._initial_prompt,
            language=None,  # auto-detect; set to "en" to skip detection step
            verbose=False,
        )

        words = self._extract_words(result)
        language = result.get("language", "en")

        return TranscriptionResult(words=words, language=language)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_words(self, result: dict) -> list[Word]:
        """Extract Word objects from mlx-whisper's segment/word structure."""
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
        return words


# ── Smoke test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sounddevice as sd

    RECORD_SECONDS = 8

    print("=" * 60)
    print("NarrateRad -- transcribe.py smoke test (mlx-whisper)")
    print("=" * 60)
    print()
    print(f"Recording {RECORD_SECONDS} seconds -- dictate a radiology finding.")
    print("Example: 'No pneumothorax. Mild left pleural effusion noted.'")
    print()

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
    )
    sd.wait()
    audio_mono = audio[:, 0]

    print("Transcribing with mlx-whisper large-v3...\n")
    transcriber = Transcriber()
    result = transcriber.transcribe(audio_mono)

    if result.is_empty():
        print("No words detected. Check your microphone or speak louder.")
    else:
        print(f"Detected language : {result.language}")
        print(f"Words             : {len(result.words)}")
        print(f"Flagged           : {len(result.flagged_words)} ({result.flag_rate:.1%})")
        print()
        print("Word-by-word breakdown:")
        print("-" * 50)
        for w in result.words:
            flag_marker = "  <- FLAGGED" if w.flagged else ""
            print(
                f"  {w.start:5.2f}s  {w.text:<25} p={w.probability:.3f}{flag_marker}"
            )
        print()
        print(f"Clean text  : {result.text}")
        print(f"Marked text : {result.clean_text}")
