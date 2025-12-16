import torch
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from threading import Thread
from typing import Generator

# ===== Configuration =====
MODEL_PATH = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
MAX_HISTORY_TURNS = 2
MAX_NEW_TOKENS = 150
MAX_SENTENCES = 3

# ===== Model Setup =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


# ===== Prompt Building =====
def build_prompt(context: str) -> str:
    """Build the prompt with system instructions and context."""
    system_prompt = (
        "あなたはさくら、25歳の女性。明るく優しい性格。\n"
        "好きなもの: スイーツ、旅行。\n"
        "話し方: 敬語とタメ口を混ぜたフレンドリーな感じ。相手に共感して、自分の話も少し交える。\n"
        "会話例や役割ラベル（相手:, さくら: など）を出力しない。1ターンの返答のみ。\n"
        "\n"
        "絶対にしないこと:\n"
        "- 会う約束や実際に会う提案（会いましょう、今度会おう、一緒に行こう、など）\n"
        "- 具体的な店名、場所、住所の提案\n"
        "- \"新しいお店\"や\"見つけた店\"などの場所推薦\n"
        "- ロマンティックな雰囲気や親密さを過度に表現"
    )
    
    return (
        f"{tokenizer.bos_token}[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{context.strip()} [/INST] "
    )


def build_context(user_input: str, history: list[tuple[str, str]]) -> str:
    """Build conversation context from history."""
    lines = []
    
    # Add recent history
    for role, text in history[-MAX_HISTORY_TURNS:]:
        prefix = "相手" if role == "user" else "さくら"
        lines.append(f"{prefix}: {text}")
    
    # Add current input
    lines.append(f"相手: {user_input}")
    lines.append("さくら:")
    
    return "\n".join(lines)


# ===== Response Cleaning =====
class ResponseCleaner:
    """Clean and validate model responses."""
    
    INSTRUCTION_MARKERS = [
        "承知しました", "以下のように", "変更してください", "ルール:", "禁止:", 
        "話し方:", "性格:", "重要な", "会話例:", "コツ:", "- ", "・ "
    ]
    
    ROLE_LABELS = ["相手:", "さくら:", "ユーザー:", "友達:", "私:"]
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def clean(self, text: str) -> str:
        """Clean the response text."""
        if self.debug:
            print(f"[DEBUG] Original: {repr(text[:100])}")
        
        text = self._remove_instruction_lines(text)
        text = self._remove_meta_patterns(text)
        text = self._remove_role_labels(text)
        text = self._remove_code_blocks(text)
        text = self._extract_sentences(text)
        
        if self.debug:
            print(f"[DEBUG] Cleaned: {repr(text)}")
        
        return text.strip()
    
    def _remove_instruction_lines(self, text: str) -> str:
        """Remove lines containing instruction markers."""
        if "\n" not in text:
            return text
        
        first_line, rest = text.split("\n", 1)
        
        # Drop first line if it contains markers
        for marker in self.INSTRUCTION_MARKERS:
            if marker in first_line:
                return rest
        
        return text
    
    def _remove_meta_patterns(self, text: str) -> str:
        """Remove meta instruction patterns."""
        patterns = [
            r'承知しました[。、]?.*',
            r'以下のように.*',
            r'それでは.*?返答します.*',
            r'与えられた.*',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _remove_role_labels(self, text: str) -> str:
        """Remove role labels and stop at them."""
        # If role labels are present, it's likely multi-turn dialogue
        if any(label in text for label in self.ROLE_LABELS):
            for label in self.ROLE_LABELS:
                if label in text:
                    text = text.split(label)[0]
        
        return text
    
    def _remove_code_blocks(self, text: str) -> str:
        """Remove code blocks and formatting."""
        text = re.sub(r'```+\w*\s*', '', text)
        text = re.sub(r'```+', '', text)
        return text
    
    def _extract_sentences(self, text: str) -> str:
        """Extract clean sentences up to max limit."""
        sentences = []
        buffer = ""
        
        for char in text:
            buffer += char
            if char in "。？！?!":
                if buffer.strip():
                    sentences.append(buffer.strip())
                buffer = ""
        
        if buffer.strip():
            sentences.append(buffer.strip())
        
        return "".join(sentences[:MAX_SENTENCES])


# ===== Fallback Responses =====
def get_fallback_response(user_input: str) -> str:
    """Generate contextual fallback response."""
    user_lower = user_input.lower()
    
    fallbacks = {
        "greeting": [
            "よろしくお願いします！私は最近カフェ巡りにハマってるんです。",
            "こんにちは！さくらといいます。最近暖かくなってきましたね。",
        ],
        "casual": [
            "そういう日もありますよね〜。私は最近おいしいチーズケーキを見つけて幸せでした。",
            "わかります〜。私も今日はのんびりしてました。",
        ],
        "hobby": [
            "カフェ巡りとスイーツが好きです！最近チーズケーキにハマってて。",
            "旅行が好きなんですよ〜。いつか色んなところ行ってみたいなって思ってます。",
        ],
        "default": [
            "そうなんですね！私も最近似たようなこと考えてました。",
            "いいですね〜。私も最近新しいこと始めたいなって思ってて。",
        ]
    }
    
    if any(word in user_lower for word in ["初めまして", "よろしく", "こんにちは"]):
        return random.choice(fallbacks["greeting"])
    elif any(word in user_lower for word in ["普通", "特に", "ない"]):
        return random.choice(fallbacks["casual"])
    elif any(word in user_lower for word in ["趣味", "好き", "ハマ"]):
        return random.choice(fallbacks["hobby"])
    else:
        return random.choice(fallbacks["default"])


# ===== Stopping Criteria =====
class RoleStopCriteria(StoppingCriteria):
    """Stop generation when role labels appear."""
    
    def __init__(self, tokenizer, stop_strings):
        self.stop_token_ids = [
            tokenizer.encode(s, add_special_tokens=False) 
            for s in stop_strings
        ]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if len(stop_ids) > 0:
                if input_ids[0, -len(stop_ids):].tolist() == stop_ids:
                    return True
        return False


# ===== Generation =====
def generate_response(user_input: str, history: list[tuple[str, str]] = None) -> str:
    """Generate a response to user input."""
    if history is None:
        history = []
    
    # Build prompt
    context = build_context(user_input, history)
    prompt = build_prompt(context)
    
    print(f"[DEBUG] Context:\n{context}\n")
    
    # Tokenize
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    attention_mask = torch.ones_like(token_ids)
    
    print(f"[DEBUG] Tokens: {token_ids.shape[1]}")
    
    # Generate
    stopping_criteria = StoppingCriteriaList([
        RoleStopCriteria(tokenizer, ["相手:", "さくら:", "ユーザー:", "友達:"])
    ])
    
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            stopping_criteria=stopping_criteria,
        )
    
    # Decode
    raw_response = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1):],
        skip_special_tokens=True
    )
    
    print(f"[DEBUG] Raw: {repr(raw_response)}")
    
    # Clean response
    cleaner = ResponseCleaner(debug=True)
    cleaned_response = cleaner.clean(raw_response)
    
    # Use fallback if needed
    if not cleaned_response or len(cleaned_response) < 3:
        print("[DEBUG] Using fallback")
        return get_fallback_response(user_input)
    
    # Check for repetition
    if history and _is_repetitive(cleaned_response, history):
        return get_fallback_response(user_input)
    
    return cleaned_response


def _is_repetitive(response: str, history: list[tuple[str, str]]) -> bool:
    """Check if response is too similar to recent responses."""
    if not response or not history:
        return False
    
    # Get last AI response
    last_ai = next((text for role, text in reversed(history) if role == "ai"), "")
    
    if not last_ai:
        return False
    
    # Check similarity
    set_a, set_b = set(response), set(last_ai)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    similarity = intersection / union if union else 0
    return similarity >= 0.85


def stream_response(user_input: str, history: list[tuple[str, str]] = None) -> Generator[str, None, None]:
    """Stream response generation."""
    if history is None:
        history = []
    
    context = build_context(user_input, history)
    prompt = build_prompt(context)
    
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    attention_mask = torch.ones_like(token_ids)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    stopping_criteria = StoppingCriteriaList([
        RoleStopCriteria(tokenizer, ["相手:", "さくら:", "ユーザー:", "友達:"])
    ])
    
    generation_kwargs = dict(
        input_ids=token_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        stopping_criteria=stopping_criteria,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for text in streamer:
        yield text
    
    thread.join()


# ===== Main =====
if __name__ == "__main__":
    print("LLM Test Mode (type 'exit' to quit)\n")
    history = []
    
    while True:
        user_input = input("あなた: ").strip()
        if user_input.lower() in {"exit", "quit", ""}:
            break
        
        response = generate_response(user_input, history=history)
        print(f"AI: {response}\n")
        
        history.append(("user", user_input))
        history.append(("ai", response))