"""
ASR (Automatic Speech Recognition) module.
Provides transcription using Kotoba Whisper and ReazonSpeech k2.
"""
import io
import os
import tempfile
import time
import numpy as np
from scipy.io.wavfile import write as wav_write

# Config
SAMPLE_RATE = 16000
PAD_SECONDS = 0.3
MAX_AUDIO_DURATION = 60.0  # Increased from 30s to 60s for longer sentences
_KOTOBA_ID = "kotoba-tech/kotoba-whisper-v2.2"

# Lazy globals
_kotoba_asr = None
_k2_model = None


def _load_kotoba_asr():
    """Load / cache the Kotoba Whisper HF pipeline."""
    global _kotoba_asr
    if _kotoba_asr is not None:
        return _kotoba_asr

    print("[voice_input] Loading Kotoba Whisper model...", flush=True)
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    _kotoba_asr = pipeline(
        "automatic-speech-recognition",
        model=_KOTOBA_ID,
        device=device,
    )
    print("[voice_input] ✓ Kotoba model loaded", flush=True)
    return _kotoba_asr


def _load_reazonspeech_k2():
    """Load / cache the ReazonSpeech k2-v2 model."""
    global _k2_model
    if _k2_model is not None:
        return _k2_model
    
    print("[voice_input] Loading ReazonSpeech k2 model...", flush=True)
    from reazonspeech.k2.asr import load_model
    _k2_model = load_model()
    print("[voice_input] ✓ ReazonSpeech k2 model loaded", flush=True)
    return _k2_model




def save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wav_write(path, sample_rate, audio_int16)


def _pad_audio_array(audio: np.ndarray, sr: int) -> np.ndarray:
    if PAD_SECONDS <= 0:
        return audio
    pad_samples = int(sr * PAD_SECONDS)
    pad = np.zeros(pad_samples, dtype=audio.dtype)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.concatenate([pad, audio, pad], axis=0)


def _ensure_16k(audio: np.ndarray, sr: int):
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio, sr


def transcribe_numpy_kotoba(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    hotwords: list[str] | None = None,
) -> str:
    """Transcribe a numpy array with Kotoba Whisper."""
    if audio.size == 0:
        return ""

    max_samples = int(sample_rate * MAX_AUDIO_DURATION)
    original_samples = audio.shape[0]
    if audio.shape[0] > max_samples:
        print(f"[voice_input] Warning: audio truncated: {original_samples} samples ({original_samples/sample_rate:.2f}s) -> {max_samples} samples ({MAX_AUDIO_DURATION:.1f}s)", flush=True)
        audio = audio[:max_samples]
    else:
        print(f"[voice_input] Audio accepted: {original_samples} samples ({original_samples/sample_rate:.2f}s)", flush=True)

    audio, sample_rate = _ensure_16k(audio, sample_rate)

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        save_wav(tmp, audio, sample_rate=sample_rate)
        asr = _load_kotoba_asr()

        # Optional biasing via initial prompt (hotwords)
        generate_kwargs = None
        if hotwords:
            try:
                # Join names; tokenizer builds prompt IDs for Whisper
                prompt_txt = "、".join(hotwords)
                # Many Whisper pipelines expose tokenizer.get_prompt_ids
                prompt_ids = asr.tokenizer.get_prompt_ids(prompt_txt, language="ja")
                generate_kwargs = {"prompt_ids": prompt_ids}
            except Exception:
                # Fallback: no bias if tokenizer doesn't support prompt ids
                generate_kwargs = None

        res = asr(tmp, generate_kwargs=generate_kwargs) if generate_kwargs else asr(tmp)
        if isinstance(res, dict):
            return res.get("text", "").strip()
        if isinstance(res, list) and res:
            return res[0].get("text", "").strip()
        return str(res)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def transcribe_numpy_k2(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Transcribe a numpy array with ReazonSpeech k2."""
    if audio.size == 0:
        return ""

    max_samples = int(sample_rate * MAX_AUDIO_DURATION)
    original_samples = audio.shape[0]
    if audio.shape[0] > max_samples:
        print(f"[voice_input] Warning: audio truncated: {original_samples} samples ({original_samples/sample_rate:.2f}s) -> {max_samples} samples ({MAX_AUDIO_DURATION:.1f}s)", flush=True)
        audio = audio[:max_samples]
    else:
        print(f"[voice_input] Audio accepted: {original_samples} samples ({original_samples/sample_rate:.2f}s)", flush=True)

    audio, sample_rate = _ensure_16k(audio, sample_rate)

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        save_wav(tmp, audio, sample_rate=sample_rate)
        from reazonspeech.k2.asr import audio_from_path, transcribe
        model = _load_reazonspeech_k2()
        audio_obj = audio_from_path(tmp)
        ret = transcribe(model, audio_obj)
        return getattr(ret, "text", str(ret)).strip()
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def transcribe_wav_bytes(data: bytes, engine: str = "reazonspeech-k2") -> tuple[str, float]:
    """Transcribe a WAV byte stream. Returns (text, seconds)."""
    import soundfile as sf

    # Validate WAV data integrity
    if len(data) < 44:  # Minimum WAV header size
        print(f"[voice_input] ❌ WAV data too small ({len(data)} bytes, need ≥44)", flush=True)
        return "", 0.0
    
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    audio = np.array(audio, dtype=np.float32)
    
    raw_duration = len(audio) / sr
    print(f"[voice_input] Raw WAV: {len(audio):,} samples, {sr}Hz, {raw_duration:.2f}s", flush=True)
    
    audio = _pad_audio_array(audio, sr)
    padded_duration = len(audio) / sr
    if padded_duration != raw_duration:
        print(f"[voice_input] After padding: {len(audio):,} samples, {padded_duration:.2f}s", flush=True)
    
    t0 = time.perf_counter()
    if engine == "kotoba":
        text = transcribe_numpy_kotoba(audio, sample_rate=sr)
    else:
        text = transcribe_numpy_k2(audio, sample_rate=sr)
    
    return text, time.perf_counter() - t0




def transcribe_wav_bytes_hotwords(
    data: bytes,
    hotwords: list[str],
    engine: str = "kotoba",
) -> tuple[str, float]:
    """
    Transcribe WAV bytes with optional hotword biasing (Kotoba only).
    Returns (text, seconds).
    """
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    audio = np.array(audio, dtype=np.float32)
    audio = _pad_audio_array(audio, sr)

    t0 = time.perf_counter()
    if engine == "kotoba":
        text = transcribe_numpy_kotoba(audio, sample_rate=sr, hotwords=hotwords or None)
    else:
        # k2 doesn't expose hotword biasing here; fall back to normal
        text = transcribe_numpy_k2(audio, sample_rate=sr)
    return text, time.perf_counter() - t0




