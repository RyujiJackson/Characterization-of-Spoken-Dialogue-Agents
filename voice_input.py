import queue
import sys
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import whisper
from scipy.io.wavfile import write as wav_write

# Load Whisper model once (change to "base" / "small" / "medium" / "large" if you like)
# "tiny" is fastest but least accurate.
_MODEL_NAME = "base"
_model = whisper.load_model(_MODEL_NAME)

# Audio settings
SAMPLE_RATE = 16000  # Whisper prefers 16k
CHANNELS = 1
DTYPE = "float32"


class VoiceRecorder:
    """
    Simple blocking recorder:
    - start_recording()
    - stop_recording() -> returns numpy array of audio samples
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._recording = False

    def _callback(self, indata, frames, time_info, status):  # noqa: D401
        """Sounddevice callback: push chunks into queue."""
        if status:
            print(f"[voice_input] Stream status: {status}", file=sys.stderr)
        if self._recording:
            # Copy to avoid referencing the same buffer
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

        # Collect queued chunks
        chunks = []
        while not self._q.empty():
            chunks.append(self._q.get())

        if not chunks:
            return np.empty((0,), dtype=DTYPE)

        audio = np.concatenate(chunks, axis=0).flatten()
        return audio


def save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save audio (float32 -1..1) to WAV file."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    # Convert to int16 for WAV
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wav_write(path, sample_rate, audio_int16)


def transcribe_audio_array(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe a numpy audio array with Whisper.
    Returns recognized text (or empty string on failure).
    """
    if audio.size == 0:
        return ""

    # Whisper expects 16k float32 mono
    if sample_rate != 16000:
        # Resample if needed
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    result = _model.transcribe(audio, language=language, fp16=False)
    return result.get("text", "").strip()


def record_and_transcribe(
    max_duration: float = 10.0,
    language: Optional[str] = None,
    save_debug_wav: Optional[str] = None,
) -> str:
    """
    High-level helper:
    - records up to `max_duration` seconds from mic
    - transcribes with Whisper
    - returns recognized text
    """
    recorder = VoiceRecorder()
    print(f"[voice_input] Recording... (up to {max_duration} seconds)")
    recorder.start_recording()

    # Simple blocking sleep; you can replace with key-press logic if you want
    time.sleep(max_duration)

    audio = recorder.stop_recording()
    print(f"[voice_input] Recording stopped. Collected {len(audio)} samples.")

    if save_debug_wav is not None:
        save_wav(save_debug_wav, audio)
        print(f"[voice_input] Saved debug WAV: {save_debug_wav}")

    text = transcribe_audio_array(audio, sample_rate=SAMPLE_RATE, language=language)
    print(f"[voice_input] Transcription: {text!r}")
    return text


def record_until_enter(language: Optional[str] = None) -> str:
    """
    Alternative usage:
    - Start recording
    - Press ENTER in terminal to stop
    """
    recorder = VoiceRecorder()
    print("[voice_input] Press ENTER to start recording.")
    input()
    print("[voice_input] Recording... Press ENTER again to stop.")
    recorder.start_recording()

    # Wait for ENTER in a separate thread
    stop_flag = {"stop": False}

    def _wait_for_enter():
        input()
        stop_flag["stop"] = True

    t = threading.Thread(target=_wait_for_enter, daemon=True)
    t.start()

    while not stop_flag["stop"]:
        time.sleep(0.1)

    audio = recorder.stop_recording()
    print(f"[voice_input] Recording stopped. Collected {len(audio)} samples.")
    text = transcribe_audio_array(audio, sample_rate=SAMPLE_RATE, language=language)
    print(f"[voice_input] Transcription: {text!r}")
    return text


if __name__ == "__main__":
    # Simple CLI demo
    # Example: python voice_input.py
    text = record_and_transcribe(max_duration=5.0, language=None, save_debug_wav="last_input.wav")
    print("Final text:", text)