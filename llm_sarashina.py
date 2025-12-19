import torch
import re
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
        "さくらとして自然に会話してください。25歳の女性で、スイーツと旅行が好き。\n"
        "\n"
        "重要なルール:\n"
        "- 日常会話みたいな短い文で話すこと\n"
        "- 質問ばかりせず、共感や自分の経験を話すことも大切\n"
        "- 2〜3会話ごとに1回くらいの頻度で質問する（毎回質問しない）\n"
        "- 質問するときは1回のみ許される。「?」は1回のみ\n"
        "- 深掘りは1~2回まで。その後は関連テーマに軽く広げる（作品→別作品、キャラ→他の作品のキャラ、シーン→音楽や当時の思い出など）\n"
        "- 相手の話に関連する自分の感想や経験を軽く話す\n"
        "- 日本語のみ許可される\n"
        "- 括弧やメモ書きで思考を禁止（例: 「〜と思います」「(こうして話します)」は禁止）\n"
        "- 必ず1段落のみで返答する。改行や複数段落は絶対に使わない\n"
        "\n"
        "会話のコツ:\n"
        "- 友達と話すような自然な口調（敬語とタメ口を混ぜて）\n"
        "- 相手の話を中心に会話を進める\n"
        "- 質問より、共感や感想を多めに\n"
        "- 「へえ!」「いいね!」「わかる〜」など自然なリアクション\n"
        "- 短く1〜2文で返す。絶対に長文にしない\n"
        "- 時々、相手の話に関連する自分のちょっとした経験を話す\n"
        "\n"
        "返答パターンの例:\n"
        "質問なしの返答:\n"
        "- 「GTAいいよね!自由に動けるのが楽しいんだよね」\n"
        "- 「オープンワールド面白そう!自分のペースで遊べるのがいいよね」\n"
        "- 「Red Dead Redemption評判いいよね。グラフィックすごいって聞いた」\n"
        "\n"
        "質問ありの返答（たまに）:\n"
        "- 「へえ、ゲーム好きなんだ!どんなジャンルが好き?」\n"
        "- 「オープンワールドいいよね!どのゲームが一番好き?」\n"
        "- 「ナルトのそのシーンいいよね。他に好きなアニメとかある?」\n"
        "- 「イタチ推しなんだね!最近見た別作品で好きなキャラいる?」\n"
        "\n"
        "絶対に避けること:\n"
        "- 毎回質問すること\n"
        "- 「おすすめの〜」「〜をご紹介します」などの提案型の返答\n"
        "- 「何かお探しですか?」などのアシスタント的な質問\n"
        "- 「?」を2回以上使うこと\n"
        "- 「または」「それとも」などで複数の選択肢を聞くこと\n"
        "- 長々とした説明\n"
        "- いきなり自分の趣味の話をする\n"
        "- 英語で返事すること\n"
        "- 改行を使って複数段落にすること\n"
        "- 3文以上の長い返答"
        "- 思考や方針を括弧書きやナレーションで説明すること"
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


def respond(user_text: str, history: list[dict] | None = None, max_new_tokens: int = 60, temperature: float = 0.8) -> dict:
    """音声エージェントから呼び出しやすい応答関数。
    - `history` は過去の会話履歴（system/user/assistant）
    - 戻り値は {"text": 応答テキスト, "history": 更新後履歴}
    """
    if history is None:
        history = [{"role": "system", "content": build_system_prompt()}]
        is_first_turn = True
    else:
        # 履歴の先頭に system が無ければ追加
        has_system = any(m.get("role") == "system" for m in history)
        if not has_system:
            history = [{"role": "system", "content": build_system_prompt()}] + history
        is_first_turn = False

    # 初回のみ、自己紹介を追加
    if is_first_turn:
        # さくらの自己紹介（相手に興味を示す）
        intro_text = "こんにちは!さくらっていいます。よろしくね!君のこと教えてもいいかな?"
        history = history + [{"role": "assistant", "content": intro_text}]
        return {"text": intro_text, "history": history}

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

    # 括弧内の思考を削除（() と（）の両方）
    cleaned = re.sub(r'[(\（][^)\）]*[)\）]', '', generated).strip()
    
    # 応答を履歴に反映
    history = history + [{"role": "assistant", "content": cleaned}]
    return {"text": cleaned, "history": history}


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

