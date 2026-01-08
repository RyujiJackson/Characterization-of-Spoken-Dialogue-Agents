#!/usr/bin/env python3
"""
Voice client with automatic endpointing (VAD) and SSH transport.

- Listens on default input device at 16kHz mono
- Detects utterance start/stop using WebRTC VAD (no Enter key)
- Sends each utterance to the remote server over SSH using existing
  length-prefixed WAV protocol (main.py --mode conversation)
- Plays TTS audio chunks streamed back from server

Usage examples:
  python voice_client_vad.py \
    --server klab.tut \
    --remote-python /home/ryuu/Ryu/Research/.venv/bin/python \
    --remote-script /home/ryuu/Ryu/Research/main.py \
    --engine kotoba

Press Ctrl+C to quit.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import select
import termios
import tty


SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20  # 10, 20, or 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per frame
SAMPLE_DTYPE = np.int16


@dataclass
class VADConfig:
    aggressiveness: int = 2        # 0-3 (3 most aggressive)
    min_speech_ms: int = 200       # require some speech to trigger (debounce)
    pre_roll_ms: int = 200         # include audio before trigger
    max_utterance_s: float = 15.0  # safety cap per utterance
    silence_end_ms: int = 800      # stop after this much trailing silence


class VADRecorder:
    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.aggressiveness)
        self._stream: sd.RawInputStream | None = None
        self._lock = threading.Lock()
        self._listening_enabled = True  # pause mic during playback if desired

    def _open_stream(self):
        if self._stream is None:
            self._stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='int16',
                blocksize=FRAME_SIZE,  # deliver ~20ms frames
            )
            self._stream.start()

    def set_listening(self, enabled: bool):
        with self._lock:
            self._listening_enabled = enabled

    def record_utterance(self) -> bytes | None:
        """Block until an utterance is detected and ended. Return WAV bytes."""
        self._open_stream()

        pre_roll_frames = int(self.cfg.pre_roll_ms / FRAME_MS)
        min_speech_frames = int(self.cfg.min_speech_ms / FRAME_MS)
        silence_end_frames = int(self.cfg.silence_end_ms / FRAME_MS)

        # Ring buffer for pre-roll
        preroll: list[bytes] = []

        # State
        triggered = False
        voiced_count = 0
        unvoiced_count = 0
        utter_frames: list[bytes] = []
        start_t = time.perf_counter()

        while True:
            with self._lock:
                listening_now = self._listening_enabled

            try:
                frame = self._stream.read(FRAME_SIZE)[0] if listening_now else b"\x00" * (FRAME_SIZE * 2)
            except Exception:
                # Input glitch; wait briefly and continue
                time.sleep(FRAME_MS / 1000.0)
                continue

            is_speech = False
            if listening_now:
                try:
                    is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    is_speech = False

            if not triggered:
                preroll.append(frame)
                if len(preroll) > pre_roll_frames:
                    preroll.pop(0)

                voiced_count = (voiced_count + 1) if is_speech else 0
                if voiced_count >= min_speech_frames:
                    triggered = True
                    utter_frames.extend(preroll)
                    preroll.clear()
                    # print("[VAD] Triggered")
            else:
                utter_frames.append(frame)
                if is_speech:
                    unvoiced_count = 0
                else:
                    unvoiced_count += 1

                # End if enough trailing silence
                if unvoiced_count >= silence_end_frames:
                    break

                # Safety max duration
                if time.perf_counter() - start_t > self.cfg.max_utterance_s:
                    break

        if not utter_frames:
            return None

        # Convert collected PCM frames to WAV bytes
        pcm = b"".join(utter_frames)
        data = np.frombuffer(pcm, dtype=SAMPLE_DTYPE).astype(np.float32) / 32768.0
        buf = io.BytesIO()
        sf.write(buf, data, samplerate=SAMPLE_RATE, format='WAV')
        buf.seek(0)
        return buf.read()


def play_audio_bytes(audio_b64: str):
    try:
        audio_bytes = base64.b64decode(audio_b64)
        buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buffer, dtype='float32')
        sd.play(audio, samplerate=sr)
        sd.wait()
    except Exception as e:
        print(f"⚠️  Audio playback error: {e}")


def make_ssh_cmd(server: str, remote_python: str, remote_script: str, engine: str) -> list[str]:
    return [
        "ssh", "-T",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        server,
        f"{remote_python} {remote_script} --mode conversation --engine {engine}",
    ]


def send_wav(proc: subprocess.Popen, wav_data: bytes) -> bool:
    try:
        if proc.stdin.closed:
            return False
        proc.stdin.write(struct.pack(">I", len(wav_data)))
        proc.stdin.flush()
        # chunked write
        for i in range(0, len(wav_data), 65536):
            proc.stdin.write(wav_data[i:i+65536])
        proc.stdin.flush()
        return True
    except Exception as e:
        print(f"❌ send error: {e}")
        return False


def reader_thread_fn(proc: subprocess.Popen, playback_flag: threading.Event, ready_flag: threading.Event):
    # Decode server stdout lines and play AUDIO_CHUNKs
    while True:
        line_bytes = proc.stdout.readline()
        if not line_bytes:
            break
        line = line_bytes.decode('utf-8', errors='replace').strip()
        if not line:
            continue

        # Server readiness
        if line.startswith("[conversation] READY"):
            ready_flag.set()
            print("✅ Server ready")
            continue

        # Streamed TTS audio
        if line.startswith("[AUDIO_CHUNK_"):
            # While playing, mute mic listening to avoid feedback
            playback_flag.set()
            try:
                b64 = line.split("]", 1)[-1].strip()
                play_audio_bytes(b64)
            finally:
                playback_flag.clear()
            continue

        # Other informational lines
        print(line)


def record_until_enter(max_seconds: float, mute_flag: threading.Event) -> bytes | None:
    """Manual mode: press Enter to stop recording."""
    print("\n🎤 Recording... press ENTER to stop.")
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=FRAME_SIZE,
    )
    stream.start()

    frames: list[bytes] = []
    start_t = time.perf_counter()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            # stop on Enter
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch in ('\r', '\n'):
                    break

            # capture a frame
            try:
                if mute_flag.is_set():
                    frame = b"\x00" * (FRAME_SIZE * 2)
                else:
                    frame = stream.read(FRAME_SIZE)[0]
                frames.append(frame)
            except Exception:
                time.sleep(FRAME_MS / 1000.0)

            if time.perf_counter() - start_t > max_seconds:
                print("\n⏱️  Max recording time reached.")
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        try:
            stream.stop(); stream.close()
        except Exception:
            pass

    if not frames:
        return None
    pcm = b"".join(frames)
    data = np.frombuffer(pcm, dtype=SAMPLE_DTYPE).astype(np.float32) / 32768.0
    buf = io.BytesIO()
    sf.write(buf, data, samplerate=SAMPLE_RATE, format='WAV')
    buf.seek(0)
    return buf.read()


def main():
    parser = argparse.ArgumentParser(description="VAD-based voice client (no button press)")
    parser.add_argument("--server", default=os.environ.get("VOICE_SERVER", "klab.tut"))
    parser.add_argument("--engine", choices=["kotoba", "reazonspeech-k2"], default="kotoba")
    parser.add_argument("--remote-python", default=os.environ.get("REMOTE_PY", "/home/ryuu/Ryu/Research/.venv/bin/python"))
    parser.add_argument("--remote-script", default=os.environ.get("REMOTE_SCRIPT", "/home/ryuu/Ryu/Research/main.py"))
    parser.add_argument("--vad-aggr", type=int, default=2, help="VAD aggressiveness 0-3")
    parser.add_argument("--silence-ms", type=int, default=800, help="Stop after this trailing silence")
    parser.add_argument("--max-utterance-s", type=float, default=15.0)
    # Toggle: auto endpointing on/off
    try:
        from argparse import BooleanOptionalAction  # py311+
        parser.add_argument("--auto-vad", action=BooleanOptionalAction, default=True, help="Enable automatic endpointing via VAD")
    except Exception:
        parser.add_argument("--auto-vad", action="store_true", default=True, help="Enable automatic endpointing via VAD")
        parser.add_argument("--no-auto-vad", dest="auto_vad", action="store_false")
    args = parser.parse_args()

    print("=" * 50)
    print("🗣️  VAD Voice Client (no button press)")
    print("=" * 50)
    print(f"🖥️  Server: {args.server}")
    print(f"⚙️  Engine: {args.engine}")
    print("🎤 Speak when ready. Ctrl+C to quit.\n")

    # Test SSH
    test_cmd = ["ssh", "-o", "ConnectTimeout=5", args.server, "echo 'SSH_OK'"]
    try:
        test = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        if test.returncode != 0 or "SSH_OK" not in test.stdout:
            print("❌ SSH test failed:")
            print(test.stdout, test.stderr)
            return
    except Exception as e:
        print(f"❌ SSH test error: {e}")
        return

    ssh_cmd = make_ssh_cmd(args.server, args.remote_python, args.remote_script, args.engine)
    print(f"📡 SSH Command: ssh {args.server} '{args.remote_python} {args.remote_script} --mode conversation --engine {args.engine}'")

    try:
        proc = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return

    server_ready = threading.Event()
    playing_audio = threading.Event()
    t_reader = threading.Thread(target=reader_thread_fn, args=(proc, playing_audio, server_ready), daemon=True)
    t_reader.start()

    # Wait up to ~60s for model init
    print("⏳ Waiting for server to initialize...")
    for i in range(600):
        if server_ready.is_set():
            break
        time.sleep(0.1)
        if (i + 1) % 50 == 0:
            sys.stderr.write(f"\r   Still waiting... {(i+1)/10:.0f}s elapsed")
            sys.stderr.flush()
    sys.stderr.write("\r" + " " * 60 + "\r")
    if not server_ready.is_set():
        print("❌ Server did not become ready in time.")
        try:
            proc.terminate()
        except Exception:
            pass
        return
    print("✅ Connected. Listening...\n")

    # Recorder
    vad_cfg = VADConfig(
        aggressiveness=args.vad_aggr,
        silence_end_ms=args.silence_ms,
        max_utterance_s=args.max_utterance_s,
    )
    recorder = VADRecorder(vad_cfg)

    try:
        while True:
            # If we're currently playing TTS, keep the mic muted
            recorder.set_listening(not playing_audio.is_set())

            if args.auto_vad:
                wav = recorder.record_utterance()
            else:
                print("Press ENTER to start recording, 'q'+ENTER to quit.")
                try:
                    cmd = input().strip()
                except EOFError:
                    cmd = 'q'
                if cmd.lower() == 'q':
                    break
                # Now capture until next Enter
                wav = record_until_enter(args.max_utterance_s, playing_audio)

            if not wav or len(wav) < 2000:
                continue

            dur_est = 0.0
            try:
                a, sr = sf.read(io.BytesIO(wav), dtype='float32')
                dur_est = len(a) / float(sr)
            except Exception:
                pass
            print(f"📤 Sending utterance ({len(wav):,} bytes, ~{dur_est:.1f}s)")

            ok = send_wav(proc, wav)
            if not ok:
                print("❌ Connection lost while sending.")
                break

            # After sending, audio is handled by reader thread.

    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
    finally:
        try:
            if proc and proc.poll() is None:
                # Send zero-length frame to request clean shutdown
                try:
                    proc.stdin.write(struct.pack(">I", 0))
                    proc.stdin.flush()
                except Exception:
                    pass
                proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
