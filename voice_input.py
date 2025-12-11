import argparse
import io
import os
import queue
import sys
import tempfile
import time
import threading
from typing import Optional
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

# Config
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
_KOTOBA_ID = "kotoba-tech/kotoba-whisper-v2.2"

# Lazy globals
_kotoba_asr = None
_k2_model = None


# ========= SPINNER CLASS =========
class Spinner:
    """A simple spinner to show processing is happening."""
    
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, message: str = "Processing"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
    
    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            elapsed = time.perf_counter() - self._start_time
            frame = self.FRAMES[idx % len(self.FRAMES)]
            sys.stderr.write(f"\r{frame} {self.message}... ({elapsed:.1f}s) ")
            sys.stderr.flush()
            idx += 1
            self._stop_event.wait(0.1)
        # Clear the line
        sys.stderr.write("\r" + " " * 60 + "\r")
        sys.stderr.flush()
    
    def start(self):
        self._start_time = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def _load_kotoba_asr():
    """Load / cache the Kotoba Whisper HF pipeline."""
    global _kotoba_asr
    if _kotoba_asr is not None:
        return _kotoba_asr

    with Spinner("Loading Kotoba Whisper model"):
        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError(
                "Install transformers+torch+soundfile+librosa:\n"
                "  pip install transformers torch soundfile librosa"
            ) from e

        try:
            import torch

            device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1

        asr = pipeline(
            "automatic-speech-recognition",
            model=_KOTOBA_ID,
            device=device,
        )
        _kotoba_asr = asr
    
    print("[voice_input] ✓ Kotoba model loaded", file=sys.stderr)
    return _kotoba_asr


def _load_reazonspeech_k2():
    """
    Load / cache the ReazonSpeech k2-v2 model.
    """
    global _k2_model
    if _k2_model is not None:
        return _k2_model
    
    with Spinner("Loading ReazonSpeech k2 model"):
        try:
            from reazonspeech.k2.asr import load_model
        except ImportError as e:
            raise RuntimeError(
                "Install ReazonSpeech k2 ASR first.\n"
                "See: https://github.com/reazon-research/ReazonSpeech"
            ) from e

        _k2_model = load_model()
    
    print("[voice_input] ✓ ReazonSpeech k2 model loaded", file=sys.stderr)
    return _k2_model


class VoiceRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._recording = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[voice_input] Stream status: {status}", file=sys.stderr)
        if self._recording:
            self._q.put(indata.copy())

    def start_recording(self):
        if self._recording:
            return
        self._q = queue.Queue()
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=DTYPE,
            callback=self._callback,
        )
        self._stream.start()

    def stop_recording(self) -> np.ndarray:
        if not self._recording:
            return np.empty((0,), dtype=DTYPE)
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        chunks = []
        while not self._q.empty():
            chunks.append(self._q.get())
        if not chunks:
            return np.empty((0,), dtype=DTYPE)
        audio = np.concatenate(chunks, axis=0).flatten()
        return audio


def save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wav_write(path, sample_rate, audio_int16)


def _pad_audio_array(audio: np.ndarray, sr: int, pad_seconds: float) -> np.ndarray:
    if pad_seconds <= 0:
        return audio
    pad_samples = int(sr * pad_seconds)
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


# ==== Kotoba path ====


def transcribe_numpy_kotoba(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """
    Transcribe a numpy array by writing a temp WAV and calling Kotoba Whisper.
    """
    if audio.size == 0:
        return ""

    max_seconds = 30.0
    max_samples = int(sample_rate * max_seconds)
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


def transcribe_file_kotoba(path: str, pad_seconds: float = 0.0) -> str:
    """Transcribe a file with Kotoba Whisper."""
    try:
        import soundfile as sf
    except Exception:
        raise RuntimeError("Install soundfile: pip install soundfile")

    data, sr = sf.read(path, dtype="float32")
    audio = data
    if pad_seconds and pad_seconds > 0:
        audio = _pad_audio_array(audio, sr, pad_seconds)
    return transcribe_numpy_kotoba(audio, sample_rate=sr)


# ==== ReazonSpeech-k2 path ====


def transcribe_file_k2(path: str) -> str:
    """
    Transcribe a file with reazonspeech-k2-v2 using the official API.
    """
    from reazonspeech.k2.asr import audio_from_path, transcribe

    model = _load_reazonspeech_k2()
    audio = audio_from_path(path)
    ret = transcribe(model, audio)
    text = getattr(ret, "text", str(ret))
    return text.strip()


def transcribe_numpy_k2(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Transcribe a numpy array by saving to temp WAV and using reazonspeech.k2.asr.
    """
    if audio.size == 0:
        return ""

    max_seconds = 30.0
    max_samples = int(sample_rate * max_seconds)
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]

    audio, sample_rate = _ensure_16k(audio, sample_rate)

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        save_wav(tmp, audio, sample_rate=sample_rate)
        return transcribe_file_k2(tmp)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# ==== shared helpers ====


def record_audio(duration: float = 5.0, verbose: bool = True) -> np.ndarray:
    """Record audio from the default microphone and return the raw samples."""
    r = VoiceRecorder()
    if verbose:
        print(f"[voice_input] Recording {duration}s...")
    r.start_recording()
    time.sleep(duration)
    audio = r.stop_recording()
    if verbose:
        print(f"[voice_input] Collected {len(audio)} samples.")
    return audio


def _transcribe_with_timing(
    audio: np.ndarray,
    sample_rate: int,
    engine: str,
    show_spinner: bool = True,
) -> tuple[str, float]:
    t0 = time.perf_counter()
    
    if show_spinner:
        spinner = Spinner(f"ASR ({engine})")
        spinner.start()
    
    try:
        if engine == "kotoba":
            text = transcribe_numpy_kotoba(audio, sample_rate=sample_rate)
        elif engine == "reazonspeech-k2":
            text = transcribe_numpy_k2(audio, sample_rate=sample_rate)
        else:
            raise ValueError(f"Unknown engine: {engine}")
    finally:
        if show_spinner:
            spinner.stop()
    
    t1 = time.perf_counter()
    return text, t1 - t0


def transcribe_wav_bytes(
    data: bytes,
    engine: str = "reazonspeech-k2",
    show_spinner: bool = True,
) -> tuple[str, float]:
    """
    Transcribe a WAV byte stream (e.g., piped over SSH). Returns (text, seconds).
    """
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    audio = np.array(audio, dtype=np.float32)
    return _transcribe_with_timing(audio, sample_rate=sr, engine=engine, show_spinner=show_spinner)


def record_and_transcribe(
    duration: float = 5.0,
    save_wav_path: Optional[str] = None,
    pad_seconds: float = 0.0,
    engine: str = "kotoba",
) -> str:
    audio = record_audio(duration=duration)
    if pad_seconds and pad_seconds > 0:
        audio = _pad_audio_array(audio, SAMPLE_RATE, pad_seconds)
    if save_wav_path:
        save_wav(save_wav_path, audio)
        print(f"[voice_input] Saved debug WAV -> {save_wav_path}")
    text, elapsed = _transcribe_with_timing(audio, sample_rate=SAMPLE_RATE, engine=engine)
    print(f"[voice_input] Transcription: {text!r}")
    print(f"[voice_input] {engine} time: {elapsed:.3f}s")
    return text


def _run_compare_mode(args: argparse.Namespace):
    """Compare Kotoba vs k2 on the same audio and print runtime."""
    if args.file:
        import soundfile as sf

        data, sr = sf.read(args.file, dtype="float32")
        audio = data
    else:
        print(f"[compare] Recording mic for {args.duration}s…")
        audio = record_audio(args.duration, verbose=False)
        sr = SAMPLE_RATE

    if args.pad_seconds and args.pad_seconds > 0:
        audio = _pad_audio_array(audio, sr, args.pad_seconds)

    audio = np.array(audio, dtype=np.float32)

    print("[compare] Running Kotoba Whisper…")
    kotoba_text, kotoba_time = _transcribe_with_timing(audio.copy(), sr, "kotoba")

    print("[compare] Running ReazonSpeech-k2…")
    k2_text, k2_time = _transcribe_with_timing(audio.copy(), sr, "reazonspeech-k2")

    print("\n=== 📝 Results ===")
    print(f"Kotoba: {kotoba_text}")
    print(f"Reazon (k2): {k2_text}")

    print("\n=== ⚡ Runtime (seconds) ===")
    print(f"Kotoba Whisper:       {kotoba_time:.3f}")
    print(f"ReazonSpeech-k2:      {k2_time:.3f}")

    print("\n=== 🔍 Match? ===")
    if kotoba_text.strip() == k2_text.strip():
        print("✔ Both match exactly!")
    else:
        print("❌ Different outputs!")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe via Kotoba Whisper or ReazonSpeech-k2-v2"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Audio file to transcribe (if omitted, record from mic)",
    )
    parser.add_argument(
        "--engine",
        choices=["kotoba", "reazonspeech-k2"],
        default="kotoba",
        help="ASR engine to use (default: kotoba)",
    )
    parser.add_argument(
        "--pad-seconds",
        type=float,
        default=0.0,
        help="Pad silence (seconds) at start/end of audio before transcribing",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (when using mic)",
    )
    parser.add_argument(
        "--save-wav",
        default=None,
        help="Path to save recorded WAV for debugging (mic mode)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Kotoba vs ReazonSpeech-k2 (runtime + text)",
    )
    parser.add_argument(
        "--stdin-wav",
        action="store_true",
        help="Read a WAV stream from stdin (useful when piping mic audio over SSH)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Pass ASR output to LLM and return the response",
    )
    parser.add_argument(
        "--conversation",
        action="store_true",
        help="Enable multi-turn conversation mode with history (reads length-prefixed WAV chunks)",
    )
    return parser.parse_args()


def _run_conversation_mode(engine: str):
    """
    Multi-turn conversation mode with full history support.
    """
    import struct
    from LLM import run_llm_COSTAR

    history: list[tuple[str, str]] = []

    print("[conversation] READY", flush=True)

    while True:
        len_bytes = sys.stdin.buffer.read(4)
        if not len_bytes or len(len_bytes) < 4:  # Fixed: was len(len(bytes))
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
            print(f"[conversation] Incomplete WAV ({len(wav_data)}/{wav_len})", file=sys.stderr, flush=True)
            break

        print(f"[conversation] Received {wav_len} bytes, transcribing...", flush=True)
        
        # ASR with spinner
        try:
            text, asr_time = transcribe_wav_bytes(wav_data, engine=engine, show_spinner=True)
        except Exception as e:
            print(f"[ASR_ERROR] {e}", flush=True)
            continue

        print(f"[ASR] {text}", flush=True)
        print(f"[ASR_TIME] {asr_time:.3f}s", flush=True)

        if not text.strip():
            print("[conversation] Empty transcription, skipping LLM.", flush=True)
            print("[RESPONSE] (音声が認識できませんでした)", flush=True)
            continue

        # LLM with spinner
        try:
            spinner = Spinner("LLM generating")
            spinner.start()
            t_llm = time.perf_counter()
            llm_response = run_llm_COSTAR(text.strip(), history=history)
            llm_time = time.perf_counter() - t_llm
            spinner.stop()
        except Exception as e:
            spinner.stop()
            print(f"[LLM_ERROR] {e}", flush=True)
            continue

        print(f"[LLM_TIME] {llm_time:.3f}s", flush=True)

        history.append(("user", text.strip()))
        history.append(("ai", llm_response))

        print(f"[RESPONSE] {llm_response}", flush=True)
        print(f"[TURN] {len(history) // 2}", flush=True)


if __name__ == "__main__":
    args = _parse_args()

    # ========= CONVERSATION MODE =========
    if args.conversation:
        _run_conversation_mode(args.engine)
        sys.exit(0)

    # ========= COMPARE MODE (early exit) =========
    if args.compare:
        _run_compare_mode(args)
        sys.exit(0)

    # ========= STDIN WAV MODE =========
    if args.stdin_wav:
        print("[voice_input] Reading WAV from stdin...", file=sys.stderr)
        data = sys.stdin.buffer.read()
        if not data:
            print("[voice_input] No stdin data received.", file=sys.stderr)
            sys.exit(1)
        
        print(f"[voice_input] Received {len(data)} bytes", file=sys.stderr)
        text, elapsed = transcribe_wav_bytes(data, engine=args.engine, show_spinner=True)
        print(f"[ASR] {text}")
        print(f"[ASR_TIME] {elapsed:.3f}s")

        # If --llm flag is set, pass the transcribed text to LLM
        if args.llm and text.strip():
            from LLM import run_llm_COSTAR

            spinner = Spinner("LLM generating")
            spinner.start()
            t0 = time.perf_counter()
            llm_response = run_llm_COSTAR(text.strip())
            t1 = time.perf_counter()
            spinner.stop()
            
            print(f"[LLM] {llm_response}")
            print(f"[LLM_TIME] {t1 - t0:.3f}s")

        sys.exit(0)

    # ========= NORMAL SINGLE-ENGINE MODE =========
    if args.file:
        t0 = time.perf_counter()
        if args.engine == "kotoba":
            text = transcribe_file_kotoba(args.file, pad_seconds=args.pad_seconds)
        else:
            text = transcribe_file_k2(args.file)
        t1 = time.perf_counter()
        print(f"Final text (file, engine={args.engine}): {text}")
        print(f"[voice_input] {args.engine} time: {t1 - t0:.3f}s")
    else:
        record_and_transcribe(
            duration=args.duration,
            save_wav_path=args.save_wav,
            pad_seconds=args.pad_seconds,
            engine=args.engine,
        )
