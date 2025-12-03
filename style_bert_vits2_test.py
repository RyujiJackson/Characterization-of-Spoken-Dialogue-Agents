from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from huggingface_hub import hf_hub_download
from pathlib import Path
import simpleaudio
import soundfile as sf

# 1) Load Japanese BERT used by Style-Bert-VITS2
bert_models.load_model(
    Languages.JP,
    "ku-nlp/deberta-v2-large-japanese-char-wwm"
)
bert_models.load_tokenizer(
    Languages.JP,
    "ku-nlp/deberta-v2-large-japanese-char-wwm"
)

# 2) Download a ready-made JP model from Hugging Face (demo voice)
model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
config_file = "jvnv-F1-jp/config.json"
style_file = "jvnv-F1-jp/style_vectors.npy"

for file in [model_file, config_file, style_file]:
    print("downloading:", file)
    hf_hub_download(
        "litagin/style_bert_vits2_jvnv",
        file,
        local_dir="model_assets"
    )

assets_root = Path("model_assets")

# 3) Create TTS model instance
model = TTSModel(
    model_path=assets_root / model_file,
    config_path=assets_root / config_file,
    style_vec_path=assets_root / style_file,
    device="cpu",  # "cuda" if you installed GPU torch
)

# 4) Run inference
text = "ひつまぶし名古屋の名物だよね"
sr, audio = model.infer(
    text=text,
    language=Languages.JP,
    style="Neutral",      # e.g. "Happy", "Sad" etc. depending on model
    style_weight=5.0,     # how strong to apply style
    length=1.0,           # >1.0 slower, <1.0 faster
    pitch_scale=1.0,      # 1.1 = slightly higher pitch, etc.
    intonation_scale=1.0, # 1.1 = more exaggerated intonation
)

# 5) Play audio
wave_obj = simpleaudio.WaveObject(audio, num_channels=1, sample_rate=sr)
play_obj = wave_obj.play()
play_obj.wait_done()

# 6) Save to file
sf.write("test_sbv2.wav", audio, samplerate=sr)

print("Saved: test_sbv2.wav")
