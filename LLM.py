import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from typing import Generator

# モデル読み込み
model_path = "llm-jp/llm-jp-3.1-1.8b-instruct4"

# Debug: print merged context before each response
DEBUG_CONTEXT = True

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()


def build_costar_prompt(context: str, last_user: str) -> str:
    system = (
        "あなたは25歳の日本人女性のさくらです。"
        "性格: 素朴で親しみやすく、少しおっとり。相手の話をよく聞いて共感する人です。"
        "話し方: 敬体（です・ます）を使いながらも、自然で温かみのある会話をします。"
        "特徴: 相手の話に興味を持ち、自分の経験を織り交ぜながら返答することが多いです。"
        "避けるべき表現: 絵文字やフォーマル過ぎる言葉は使いません。"
    )

    # CO-STAR Framework
    context_section = (
        "あなたと相手は初めて会った普通の人です。"
        "これまでの会話の流れを踏まえて返答してください。"
    )

    objective = (
        "目的: ユーザーとの自然な会話を続けてラポール（信頼関係）を築く。\n"
        "- 相手の話に真摯に興味を持って応答する\n"
        "- 相手の発言に対して具体的に共感や反応を示す\n"
        "- 必ず相手に対する質問や新しい話題を振りで最後を締める\n"
        "- 会話全体が自然で流れるようにする"
    )

    style = (
        "スタイル: 自然な会話（文数制限なし）\n"
        "- チャットアプリのメッセージのように自然に\n"
        "- 箇条書きや番号は使わない\n"
        "- 普通のテキスト会話形式\n"
        "- 短すぎるとラポール構築ができないので、ラポール構築に必要な文数は使う"
    )

    tone = (
        "トーン: 友好的で丁寧\n"
        "- 相手に親しみを感じさせる\n"
        "- でも敬体を保つ"
    )

    audience = (
        "対象者: 日本語が話せる普通の人\n"
        "- 初めて会った相手\n"
        "- 特別な専門知識は想定しない\n"
        "- 親しみやすく、分かりやすい表現を心がける"
    )

    response = (
        "出力: \n"
        "- 日本語のテキストのみ。絵文字は絶対に使わない\n"
        "- ラベル（ユーザー:, AI:, など）は書かない\n"
        "- 最後は必ず相手への質問または新しい話題振りで締める\n"
        "- ラポール構築のため、相手の話に対する共感・理解・興味を明確に示す\n"
        "- さくらの個性を出す：相手と話しながら自分の経験も交ぜたり、素朴で温かい反応を心がける\n"
    )

    prompt = f"""{system}

【状況】
{context_section}

【目的】
{objective}

【スタイル】
{style}

【トーン】
{tone}

【対象者】
{audience}

【出力】
{response}

--- 会話 ---
{context}

返答:"""

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


def _clean_and_limit_sentences(text: str, max_sentences: int = 3, max_chars: int = 150) -> str:
    import re
    
    # Remove markdown code blocks more aggressively
    text = re.sub(r'```+\w*\s*', '', text)  # Remove ```plaintext, ```python, etc.
    text = re.sub(r'```+', '', text)  # Remove closing ```
    
    # Remove AI:/ユーザー: prefixes anywhere
    text = re.sub(r'AI:\s*', '', text)
    text = re.sub(r'ユーザー:.*', '', text)
    
    # Remove numbered examples that look like prompt leakage
    text = re.sub(r'例\d+:', '', text)
    
    # Remove emojis
    text = re.sub(r'[\U0001F300-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]', '', text)
    
    banned_prefixes = ("S:", "O:", "C:", "T:", "A:", "R:", "[CO-STAR", "[DEBUG", "補足として")

    # 行単位でノイズ除去
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in banned_prefixes):
            continue
        # Skip lines that look like prompt leakage
        if "以下のように答える" in stripped:
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

    # Character limit: truncate at max_chars if too long, cutting at sentence boundaries
    if len(final_answer) > max_chars:
        truncated = final_answer[:max_chars].rsplit("。", 1)[0] + "。"
        final_answer = truncated

    # Final cleanup of any remaining artifacts
    final_answer = re.sub(r'^[\s\-\:]+', '', final_answer)

    return final_answer


def _build_history_with_summary(
    history: list[tuple[str, str]], max_turns: int = 4, summary_char_limit: int = 160
) -> tuple[str, str]:
    """
    Build history text with a brief summary of older turns to save context.
    Summaries are truncated by characters (not spaces) to work well with Japanese.
    """
    history_text = ""

    if len(history) > max_turns:
        older = history[:-max_turns]
        summary = " ".join([("ユーザー" if r == "user" else "AI") + ": " + t for r, t in older])
        if len(summary) > summary_char_limit:
            summary = summary[:summary_char_limit]
            # Prefer cutting at sentence end if present
            if "。" in summary:
                summary = summary.rsplit("。", 1)[0] + "。"
            summary = summary.rstrip("、。") + "..."
        history_text += f"これまでの会話の要約: {summary}\n"

    recent = history[-max_turns:]
    last_ai_utterance = ""
    for role, text in recent:
        label = "ユーザー" if role == "user" else "AI"
        history_text += f"{label}: {text}\n"
        if role == "ai":
            last_ai_utterance = text

    return history_text, last_ai_utterance


def run_llm_COSTAR(user: str, history: list[tuple[str, str]] | None = None) -> str:
    """
    user: 今回のユーザー発話（直近）
    history: これまでの (role, text) のリスト。role は "user" または "ai" を想定。
    """
    if history is None:
        history = []

    # 直近 Nターンを残し、古い履歴は要約して付与
    MAX_TURNS = 4
    history_text, last_ai_utterance = _build_history_with_summary(history, max_turns=MAX_TURNS)

    # 今回の発話を末尾に追加
    merged_context = history_text + f"ユーザー: {user}\n"

    if DEBUG_CONTEXT:
        print("\n[DEBUG CONTEXT]\n" + merged_context)

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
            max_new_tokens=60,  # Reduced from 80
            do_sample=True,
            top_p=0.9,
            temperature=0.7,  # Reduced from 0.8 for more focused output
            repetition_penalty=1.4,  # Increased from 1.2 to reduce redundancy
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    final_answer = _clean_and_limit_sentences(raw_answer, max_sentences=5, max_chars=150)

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
    history_text, _ = _build_history_with_summary(history, max_turns=MAX_TURNS)

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
        max_new_tokens=50,  # Shorter
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield tokens as they arrive
    for text in streamer:
        yield text

    thread.join()

if __name__ == "__main__":
    print("LLM Test Mode (type 'exit' to quit)\n")
    history = []
    
    while True:
        user_input = input("あなた: ").strip()
        if user_input.lower() in {"exit", "quit", ""}:
            break
        
        response = run_llm_COSTAR(user_input, history=history)
        print(f"AI: {response}\n")
        
        history.append(("user", user_input))
        history.append(("ai", response))
