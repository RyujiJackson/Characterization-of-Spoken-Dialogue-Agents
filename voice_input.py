import argparse
import io
import os
import sys
import tempfile
import time
import numpy as np
from scipy.io.wavfile import write as wav_write

# Config
SAMPLE_RATE = 16000
PAD_SECONDS = 0.3
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


def transcribe_numpy_kotoba(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Transcribe a numpy array with Kotoba Whisper."""
    if audio.size == 0:
        return ""

    max_samples = int(sample_rate * 30.0)
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]

    audio, sample_rate = _ensure_16k(audio, sample_rate)

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        save_wav(tmp, audio, sample_rate=sample_rate)
        asr = _load_kotoba_asr()
        res = asr(tmp)
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

    max_samples = int(sample_rate * 30.0)
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]

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

    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    audio = np.array(audio, dtype=np.float32)
    audio = _pad_audio_array(audio, sr)
    
    t0 = time.perf_counter()
    if engine == "kotoba":
        text = transcribe_numpy_kotoba(audio, sample_rate=sr)
    else:
        text = transcribe_numpy_k2(audio, sample_rate=sr)
    
    return text, time.perf_counter() - t0


def _run_conversation_mode(engine: str):
    """Multi-turn conversation mode with full history support."""
    import struct
    from LLM import run_llm_COSTAR

    history: list[tuple[str, str]] = []

    print(f"[conversation] READY", flush=True)

    while True:
        len_bytes = sys.stdin.buffer.read(4)
        if not len_bytes or len(len_bytes) < 4:
            print("[conversation] Connection closed.", flush=True)
            break

        wav_len = struct.unpack(">I", len_bytes)[0]

        if wav_len == 0:
            print("[conversation] Exit signal received. Goodbye!", flush=True)
            break

        wav_data = b""
        remaining = wav_len
        while remaining > 0:
            chunk = sys.stdin.buffer.read(min(remaining, 65536))
            if not chunk:
                break
            wav_data += chunk
            remaining -= len(chunk)

        if len(wav_data) < wav_len:
            print(f"[conversation] Incomplete WAV ({len(wav_data)}/{wav_len})", flush=True)
            break

        # ASR
        print(f"[ASR_START]", flush=True)
        try:
            text, asr_time = transcribe_wav_bytes(wav_data, engine=engine)
        except Exception as e:
            print(f"[ASR_ERROR] {e}", flush=True)
            continue

        print(f"[ASR] {text}", flush=True)
        print(f"[ASR_TIME] {asr_time:.3f}s", flush=True)

        if not text.strip():
            print("[RESPONSE] (音声が認識できませんでした)", flush=True)
            continue

        # LLM
        print(f"[LLM_START]", flush=True)
        try:
            t_llm = time.perf_counter()
            llm_response = run_llm_COSTAR(text.strip(), history=history)
            llm_time = time.perf_counter() - t_llm
        except Exception as e:
            print(f"[LLM_ERROR] {e}", flush=True)
            continue

        print(f"[LLM_TIME] {llm_time:.3f}s", flush=True)

        history.append(("user", text.strip()))
        history.append(("ai", llm_response))

        print(f"[RESPONSE] {llm_response}", flush=True)
        print(f"[TURN] {len(history) // 2}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice input server")
    parser.add_argument(
        "--engine",
        choices=["kotoba", "reazonspeech-k2"],
        default="reazonspeech-k2",
    )
    parser.add_argument("--conversation", action="store_true")
    args = parser.parse_args()

    if args.conversation:
        _run_conversation_mode(args.engine)
    else:
        print("Usage: python voice_input.py --engine reazonspeech-k2 --conversation")
        sys.exit(1)
