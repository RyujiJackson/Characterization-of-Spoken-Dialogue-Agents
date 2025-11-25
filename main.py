import audeer
import audonnx
import numpy as np
import os
import soundfile as sf
from scipy.signal import resample_poly
import time  #for process time measurement

# Import functions from your scripts
#from LLM import run_llm
from LLM import run_llm_COSTAR
from TTS_run import tts_read

def main():
    print("Starting LLM + TTS Chat Application...\n")
    print("Type your message and press Enter.")
    print("Type 'exit' or 'quit' to end the chat.\n")

    # ここで会話履歴を保持
    history: list[tuple[str, str]] = []

    while True:
        user_text = input("あなた: ").strip()

        if user_text.lower() in {"exit", "quit"} or user_text == "":
            print("チャットを終了します。")
            break

        # LLM processing time measurement
        t0 = time.perf_counter()
        llm_output = run_llm_COSTAR(user_text, history=history)
        t1 = time.perf_counter()
        llm_time = t1 - t0

        print(f"AI: {llm_output}")
        print(f"[DEBUG] LLM time: {llm_time:.3f} 秒\n")

        # 履歴を更新（必ず LLM 呼び出し後に）
        history.append(("user", user_text))
        history.append(("assistant", llm_output))

        # TTS
        print("（AIの返答を音声で再生中…）")
        t2 = time.perf_counter()
        tts_read(llm_output)
        t3 = time.perf_counter()
        tts_time = t3 - t2

        print(f"[DEBUG] TTS time: {tts_time:.3f} 秒\n")

        total_time = llm_time + tts_time
        print(f"[SUMMARY] LLM: {llm_time:.3f} 秒 / TTS: {tts_time:.3f} 秒 / Total: {total_time:.3f} 秒\n")

if __name__ == "__main__":
    main()