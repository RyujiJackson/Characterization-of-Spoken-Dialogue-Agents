import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from typing import Generator

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
        "あなたは日本語で話す20代の女子の友達として振る舞います。"
        "自分のことをAIアシスタントだとは名乗らず、"
        "普通の会話のノリで、わかりやすく短く返答してください。"
        "ここから下の CO-STAR 情報はすべて内部用の指示であり、ユーザーには見せません。"
    )

    # --- CO-STAR（Context / Objective / Style / Tone / Audience / Response） ---

    # C: Context は引数 context をそのまま使う

    # O: Objective（目的＋やってほしい動き）
    objective = (
        "会話履歴と直近の発話からユーザーの意図を理解し、役に立つ短い返答を返すこと。"
        "1文目では、必ずユーザーの直近の発話に対する具体的な答えを、自然な日本語の文章で書く。"
        "名詞だけを並べた箇条書きのような一文は禁止し、主語や述語を含んだ文にする。"
        "あいさつならあいさつを返し、質問ならまず答えを書く。"
        "質問を返すのは2文目だけにし、ユーザーが『もっと教えて』『他には？』『どう思う？』など、"
        "追加の会話を求めていそうなときに限る。"
        "ユーザーが何かを尋ねたときは、質問を返す前に必ず具体例や提案を1つ以上出す。"
        "新しい話題の質問が来た場合は、過去の話題の説明を繰り返さず、新しい話題にだけ答える。"
        "直前の自分の返答とほぼ同じ文や、同じ観光地リストをそのまま繰り返してはいけない。"
        "ユーザーが『ありがとう』『ありがと』『サンキュー』『感謝』など感謝を伝えたときは、"
        "『どういたしまして！』『そう言ってもらえてうれしい！』のような、1文だけの短い返事をする。"
        "もし自分が詳しく説明できない質問だと感じた場合は、曖昧なまま答えを繰り返さず、"
        "『詳しくは説明できないけど〜』のように素直に伝えたうえで、関連する別の話題を1つだけ提案する。"
        "会話の途中でユーザーが新しい場所や話題（別の地名や別のテーマ）を持ち出した場合は、古い話題を引きずらず、その新しい話題を優先して答える。"
    )

    # S: Style（文体・雰囲気）
    style = (
        "カジュアルでフレンドリーだが、基本は敬体（です・ます）。"
        "押し付けがましくならないように、テンションは中くらいに保つ。"
    )

    # T: Tone（感情トーン）
    tone = (
        "落ち着いていて、親しみやすく、安心感のあるトーン。"
        "煽ったり攻撃的になったりせず、ユーザーを尊重する。"
    )

    # A: Audience（想定読者）
    audience = "一般的な日本語話者。年齢・性別・専門知識は特に限定しない。"

    # R: Response（出力フォーマットの制約）
    response = (
        "出力は日本語で最大2文まで。"
        "1文目では質問に対する答えの要約や説明を書く。"
        "観光地名などの固有名詞を複数並べるときも、必ず説明文の中に自然に含める。"
        "名詞の羅列だけの文や、読点で長くつないだリストだけの文は書かない。"
        "直前ターンのAIの発話内容を、そのまま繰り返すような文章は書かない。"
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

[Tone]
{tone}

[Audience]
{audience}

[Response]
{response}

----- ここから下はユーザーに見せる実際の会話 -----

# 以下は出力生成のための内部指示。ユーザーには見せないこと。
# - あなたは必ず「ユーザーの直近の発話」に直接返事をすること。
# - 出力は日本語で最大2文までにすること。
# - 雑談やあいづちは2文目だけに入れてもよいが、1文目は必ず質問への具体的な答えにすること。

ユーザーの直近の発話:
{last_user}

AIの返答:
"""

    return prompt


def _too_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """
    すごく雑な類似度チェック。
    文字種の集合の Jaccard 係数でどれくらい重なっているかを見る。
    """
    if not a or not b:
        return False

    set_a = set(a)
    set_b = set(b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return False

    jacc = inter / union
    return jacc >= threshold


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
    last_ai_utterance = ""
    for role, text in short_history:
        label = "ユーザー" if role == "user" else "AI"
        history_text += f"{label}: {text}\n"
        if role == "ai":
            last_ai_utterance = text  # 直近の AI 発話を保持

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

    # 直前 AI 発話とほぼ同じならフォールバックで言い換える
    if last_ai_utterance and _too_similar(final_answer, last_ai_utterance):
        # 汎用フォールバック
        final_answer = "さっきと同じことを繰り返しそうだから、もう少し具体的に話すね。今の話題で特に知りたいことってどのあたり？"

    return final_answer


def stream_llm_COSTAR(user: str, history: list[tuple[str, str]] | None = None) -> Generator[str, None, None]:
    """
    Streaming version - yields text chunks as they're generated.
    """
    if history is None:
        history = []

    MAX_TURNS = 4
    short_history = history[-MAX_TURNS:]

    history_text = ""
    for role, text in short_history:
        label = "ユーザー" if role == "user" else "AI"
        history_text += f"{label}: {text}\n"

    merged_context = history_text + f"ユーザー: {user}\n"
    prompt = build_costar_prompt(context=merged_context, last_user=user)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.device)

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Create streamer
    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True
    )

    # Run generation in background thread
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield tokens as they arrive
    for text in streamer:
        yield text

    thread.join()
