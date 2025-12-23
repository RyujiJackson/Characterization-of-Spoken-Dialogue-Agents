import argparse
import base64
import struct
import sys
import time
import threading
import queue

# Import functions from your scripts
from llm_sarashina import respond
from TTS_run import tts_read, tts_synthesize, tts_speak
from voice_input import transcribe_wav_bytes


def stream_and_speak(user_text: str, history: list[dict]) -> str:
    """
    Stream LLM tokens and start TTS as soon as a sentence is complete.
    Returns the full response text.
    """
    sentence_endings = {"。", "！", "？", "!", "?"}
    
    result = respond(user_text, history=history, max_new_tokens=60, temperature=0.8)
    full_response = result["text"]
    updated_history = result["history"]
    
    print("AI: ", end="", flush=True)
    
    # Split response into sentences and speak them
    buffer = ""
    for char in full_response:
        print(char, end="", flush=True)
        buffer += char
        
        for ending in sentence_endings:
            if ending in buffer:
                idx = buffer.index(ending) + 1
                sentence = buffer[:idx].strip()
                buffer = buffer[idx:]
                
                if sentence:
                    threading.Thread(
                        target=tts_speak,
                        args=(sentence,),
                        daemon=True
                    ).start()
                break
    
    # Speak any remaining text
    if buffer.strip():
        tts_speak(buffer.strip())
    
    print()  # newline after streaming
    return full_response, updated_history


def run_text_chat():
    """Text-based chat mode with streaming TTS."""
    print("Starting LLM + TTS Chat Application (Streaming Mode)...\n")
    print("Type your message and press Enter.")
    print("Type 'exit' or 'quit' to end the chat.\n")

    history = None

    while True:
        user_text = input("あなた: ").strip()

        if user_text.lower() in {"exit", "quit"} or user_text == "":
            print("チャットを終了します。")
            break

        t0 = time.perf_counter()
        llm_output, history = stream_and_speak(user_text, history)
        total_time = time.perf_counter() - t0

        print(f"[DEBUG] Total time (LLM+TTS): {total_time:.3f} 秒\n")


def run_conversation_mode(engine: str = "reazonspeech-k2"):
    """Multi-turn voice conversation mode with streaming TTS."""
    history = None
    sentence_endings = {"。", "！", "？", "!", "?"}

    print("[conversation] READY", flush=True)

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
        print("[ASR_START]", flush=True)
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

        # LLM + TTS
        print("[LLM_START]", flush=True)
        t_llm_start = time.perf_counter()
        
        try:
            result = respond(text.strip(), history=history, max_new_tokens=60, temperature=0.8)
            full_response = result["text"]
            history = result["history"]
            
            sentence_count = 0
            buffer = ""
            
            # Process response character by character for TTS chunks
            for char in full_response:
                buffer += char
                
                for ending in sentence_endings:
                    if ending in buffer:
                        idx = buffer.index(ending) + 1
                        sentence = buffer[:idx].strip()
                        buffer = buffer[idx:]
                        
                        if sentence:
                            sentence_count += 1
                            print(f"[TTS_CHUNK_{sentence_count}] {sentence}", flush=True)
                            try:
                                t_tts = time.perf_counter()
                                audio_bytes = tts_synthesize(sentence)
                                tts_time = time.perf_counter() - t_tts
                                print(f"[TTS_CHUNK_TIME_{sentence_count}] {tts_time:.3f}s", flush=True)
                                
                                audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
                                print(f"[AUDIO_CHUNK_{sentence_count}] {audio_b64}", flush=True)
                            except Exception as e:
                                print(f"[TTS_CHUNK_ERROR] {e}", flush=True)
                        break
            
            # Handle remaining buffer
            if buffer.strip():
                sentence_count += 1
                print(f"[TTS_CHUNK_{sentence_count}] {buffer.strip()}", flush=True)
                try:
                    t_tts = time.perf_counter()
                    audio_bytes = tts_synthesize(buffer.strip())
                    tts_time = time.perf_counter() - t_tts
                    print(f"[TTS_CHUNK_TIME_{sentence_count}] {tts_time:.3f}s", flush=True)
                    
                    audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
                    print(f"[AUDIO_CHUNK_{sentence_count}] {audio_b64}", flush=True)
                except Exception as e:
                    print(f"[TTS_CHUNK_ERROR] {e}", flush=True)
                    
        except Exception as e:
            print(f"[LLM_ERROR] {e}", flush=True)
            continue

        llm_time = time.perf_counter() - t_llm_start
        print(f"[LLM_TIME] {llm_time:.3f}s", flush=True)
        print(f"[RESPONSE] {full_response}", flush=True)
        print(f"[TURN] {len(history) // 2}", flush=True)
        print("[AUDIO_DONE]", flush=True)


def main():
    parser = argparse.ArgumentParser(description="LLM + TTS Chat Application")
    parser.add_argument(
        "--mode",
        choices=["text", "conversation"],
        default="text",
        help="Chat mode: 'text' for keyboard input, 'conversation' for voice input via stdin",
    )
    parser.add_argument(
        "--engine",
        choices=["kotoba", "reazonspeech-k2"],
        default="reazonspeech-k2",
        help="ASR engine for conversation mode",
    )
    args = parser.parse_args()

    if args.mode == "conversation":
        run_conversation_mode(engine=args.engine)
    else:
        run_text_chat()


if __name__ == "__main__":
    main()