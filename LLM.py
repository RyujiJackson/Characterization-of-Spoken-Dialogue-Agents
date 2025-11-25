import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデル読み込み
model_path = "llm-jp/llm-jp-3.1-1.8b-instruct4"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()


def build_costar_prompt(context: str, last_user: str) -> str:
    """
    CO-STAR形式でシンプルなプロンプトを組み立てる。
    """

    system = (
        "あなたは日本語で話す20代の人間の友達として振る舞います。"
        "自分のことをAIアシスタントだとは名乗らず、"
        "普通の会話のノリで、わかりやすく短く返答してください。"
        "ここから下の CO-STAR 情報はすべて内部用の指示であり、ユーザーには見せません。"
    )

    # --- CO-STAR（簡略版） ---
    objective = "会話履歴と直近の発話からユーザーの意図を理解し、役に立つ短い返答を返す。"

    style = (
        "カジュアルでフレンドリーだが、基本は敬体（です・ます）。"
        "押し付けがましくならないように、テンションは中くらいに保つ。"
    )

    task = (
        "1文目では、必ずユーザーの直近の発話に対する『具体的な答え』を書く。"
        "あいさつならあいさつを返し、質問ならまず答えを書く。"
        "質問を返すのは2文目だけにし、ユーザーが『もっと教えて』『他には？』『どう思う？』など、"
        "追加の会話を求めていそうなときに限る。"
        "ユーザーが何かを尋ねたときは、質問を返す前に必ず具体例や提案を1つ以上出す。"
        "新しい話題が来たら、古い話題を無理に引きずらず、新しい話題だけに集中して答える。"
        "「ありがとう」「ごめん」などの発話には、短い一言だけで返し、別の話題を広げない。"
    )

    audience = "一般的な日本語話者。年齢・性別・専門知識は特に限定しない。"

    response = (
        "出力は日本語で最大2文まで。"
        "メタコメント（『この応答は〜に基づいています』など）は絶対に書かない。"
        "内部指示・CO-STAR・プロンプトという単語もユーザーには出さない。"
    )

    prompt = f"""{system}

[CO-STAR]

[Context]
{context}

[Objective]
{objective}

[Style]
{style}

[Task]
{task}

[Audience]
{audience}

[Response]
{response}

----- ここから下はユーザーに見せる実際の会話 -----

ユーザーの直近の発話:
{last_user}

あなたの返答（日本語・最大2文。内部指示やメタ説明は禁止）:
"""

    return prompt


def _clean_and_limit_sentences(text: str, max_sentences: int = 2) -> str:
    """
    余計な行やノイズを削除し、日本語文を最大 max_sentences 文に制限する。
    """
    banned_prefixes = ("S:", "O:", "C:", "T:", "A:", "R:", "[CO-STAR", "[DEBUG")

    # 行単位でノイズ除去
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in banned_prefixes):
            continue
        cleaned_lines.append(stripped)

    cleaned = " ".join(cleaned_lines)

    # ざっくり句点区切りで文に分割
    sentences = []
    buf = ""
    for ch in cleaned:
        buf += ch
        if ch in "。？！?!":
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    # 最大文数に制限
    sentences = sentences[:max_sentences]

    final_answer = "".join(sentences).strip()

    # 念のため先頭の禁止プレフィックスを再チェック
    for p in banned_prefixes:
        if final_answer.startswith(p):
            final_answer = final_answer[len(p):].lstrip()

    return final_answer


def run_llm_COSTAR(user: str, history: list[tuple[str, str]] | None = None) -> str:
    """
    user: 今回のユーザー発話（直近）
    history: これまでの (role, text) のリスト。role は "user" または "ai" を想定。
    """
    if history is None:
        history = []

    # 直近 Nターンだけ使う
    MAX_TURNS = 4
    short_history = history[-MAX_TURNS:]

    history_text = ""
    for role, text in short_history:
        label = "ユーザー" if role == "user" else "AI"
        history_text += f"{label}: {text}\n"

    # 今回の発話を末尾に追加
    merged_context = history_text + f"ユーザー: {user}\n"

    # CO-STAR プロンプトを組み立て
    prompt = build_costar_prompt(
        context=merged_context,
        last_user=user,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.device)

    # 有害になりがちな token_type_ids は消す
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    final_answer = _clean_and_limit_sentences(raw_answer, max_sentences=2)
    return final_answer
