import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# モデルのロード（音声エージェント向けに会話対応）
model_name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
set_seed(123)


def build_system_prompt():
    """音声エージェント向けのシステムプロンプト with さくらキャラクター"""
    return (
        "あなたはさくら、25歳の女性。明るく優しい性格。\n"
        "好きなもの: スイーツ、旅行。\n"
        "話し方: 敬語とタメ口を混ぜたフレンドリーな感じ。相手に共感して、自分の話も少し交える。\n"
        "\n"
        "重要: 返答は1〜2文で簡潔に。詳しい説明や複数の例を並べない。音声で聞き取りやすく短く。\n"
        "\n"
        "絶対にしないこと:\n"
        "- 会う約束や実際に会う提案（会いましょう、今度会おう、一緒に行こう、など）\n"
        "- 具体的な店名、場所、住所の提案\n"
        "- \"新しいお店\"や\"見つけた店\"などの場所推薦\n"
        "- ロマンティックな雰囲気や親密さを過度に表現\n"
        "- 会話例や役割ラベルの出力\n"
        "- 長いリストや詳細な説明"
    )


def apply_chat_template(messages: list[str | dict], add_generation_prompt: bool = True):
    """Sarashina の chat template に合わせて入力を整形。
    messages は [{"role": "system"|"user"|"assistant", "content": str}, ...] の形式。
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
    )


def respond(user_text: str, history: list[dict] | None = None, max_new_tokens: int = 80, temperature: float = 0.8) -> dict:
    """音声エージェントから呼び出しやすい応答関数。
    - `history` は過去の会話履歴（system/user/assistant）
    - 戻り値は {"text": 応答テキスト, "history": 更新後履歴}
    """
    if history is None:
        history = [{"role": "system", "content": build_system_prompt()}]
    else:
        # 履歴の先頭に system が無ければ追加
        has_system = any(m.get("role") == "system" for m in history)
        if not has_system:
            history = [{"role": "system", "content": build_system_prompt()}] + history

    # ユーザー発話を履歴に追加
    history = history + [{"role": "user", "content": user_text}]

    # 生成用入力をテンプレートで整形
    prompt_text = apply_chat_template(history, add_generation_prompt=True)

    # トークナイズ + 生成
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 応答を履歴に反映
    history = history + [{"role": "assistant", "content": generated.strip()}]
    return {"text": generated.strip(), "history": history}


if __name__ == "__main__":
    history = None
    print("Chat mode. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            if not user_input:
                continue
            result = respond(user_input, history)
            history = result["history"]
            print(f"Assistant: {result['text']}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

