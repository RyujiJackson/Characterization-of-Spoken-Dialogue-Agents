from pathlib import Path
from typing import Optional

import sounddevice as sd  # Add this import
import soundfile as sf
from huggingface_hub import hf_hub_download

from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel

# ---------------------------------------------------------------------------
# Global TTS model instance (lazy-loaded)
# ---------------------------------------------------------------------------

_model: Optional[TTSModel] = None
_assets_root = Path("model_assets")

# Hugging Face repo and files (from style_bert_vits2_test.py reference)
_HF_REPO_ID = "litagin/style_bert_vits2_jvnv"
_MODEL_FILE = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
_CONFIG_FILE = "jvnv-F1-jp/config.json"
_STYLE_FILE = "jvnv-F1-jp/style_vectors.npy"


def _download_assets() -> None:
    """Download required model assets if they do not exist."""
    _assets_root.mkdir(parents=True, exist_ok=True)

    for file in [_MODEL_FILE, _CONFIG_FILE, _STYLE_FILE]:
        local_path = _assets_root / file
        if not local_path.exists():
            hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=file,
                local_dir=str(_assets_root),
            )


def _load_bert() -> None:
    """Load Japanese BERT model and tokenizer only once."""
    bert_models.load_model(
        Languages.JP,
        "ku-nlp/deberta-v2-large-japanese-char-wwm",
    )
    bert_models.load_tokenizer(
        Languages.JP,
        "ku-nlp/deberta-v2-large-japanese-char-wwm",
    )


def _init_model() -> None:
    """Initialize the global TTS model if it is not already loaded."""
    global _model
    if _model is not None:
        return

    _download_assets()
    _load_bert()

    _model = TTSModel(
        model_path=_assets_root / _MODEL_FILE,
        config_path=_assets_root / _CONFIG_FILE,
        style_vec_path=_assets_root / _STYLE_FILE,
        device="cpu",  # change to "cuda" if GPU is available
    )


def tts_read(
    text: str,
    *,
    style: str = "Neutral",
    style_weight: float = 5.0,
    length: float = 1.0,
    pitch_scale: float = 1.0,
    intonation_scale: float = 1.0,
    save_path: Optional[Path] = None,
) -> None:
    """
    Convert text to speech using Style-Bert-VITS2 and save it as output.wav.

    If save_path is given, save there instead.
    """
    if not text:
        return

    _init_model()
    assert _model is not None  # for type checkers

    sr, audio = _model.infer(
        text=text,
        language=Languages.JP,
        style=style,
        style_weight=style_weight,
        length=length,
        pitch_scale=pitch_scale,
        intonation_scale=intonation_scale,
    )

    # Determine output path
    if save_path is None:
        save_path = Path("output.wav")
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(save_path), audio, samplerate=sr)


def tts_speak(
    text: str,
    *,
    style: str = "Neutral",
    style_weight: float = 5.0,
    length: float = 1.0,
    pitch_scale: float = 1.0,
    intonation_scale: float = 1.0,
) -> None:
    """
    Synthesize and immediately play audio (blocking).
    """
    if not text:
        return

    _init_model()
    assert _model is not None

    sr, audio = _model.infer(
        text=text,
        language=Languages.JP,
        style=style,
        style_weight=style_weight,
        length=length,
        pitch_scale=pitch_scale,
        intonation_scale=intonation_scale,
    )

    # Play audio directly
    sd.play(audio, samplerate=sr)
    sd.wait()


def tts_synthesize(
    text: str,
    *,
    style: str = "Neutral",
    style_weight: float = 5.0,
    length: float = 1.0,
    pitch_scale: float = 1.0,
    intonation_scale: float = 1.0,
) -> bytes:
    """
    Synthesize text to speech and return WAV bytes (for streaming).
    """
    if not text:
        return b""

    _init_model()
    assert _model is not None

    sr, audio = _model.infer(
        text=text,
        language=Languages.JP,
        style=style,
        style_weight=style_weight,
        length=length,
        pitch_scale=pitch_scale,
        intonation_scale=intonation_scale,
    )

    # Convert to WAV bytes
    import io

    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=sr, format="WAV")
    buffer.seek(0)
    return buffer.read()