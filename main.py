import audeer
import audonnx
import numpy as np
import os
import soundfile as sf
from scipy.signal import resample_poly

# --- Configuration Constants ---
MODEL_URL = 'https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'
CACHE_ROOT = 'cache'
MODEL_ROOT = 'model'
TARGET_SAMPLING_RATE = 16000
GENDER_LABELS = ['female', 'male', 'child']


def load_model():
    """Loads or downloads/extracts the age/gender prediction model."""
    audeer.mkdir(CACHE_ROOT)
    audeer.mkdir(MODEL_ROOT)

    try:
        # Check if model files exist locally
        if not os.path.exists(os.path.join(MODEL_ROOT, 'model.onnx')) or \
           not os.path.exists(os.path.join(MODEL_ROOT, 'config.yaml')):
            print("Model files not found locally. Attempting to download and extract...")
            archive_path = audeer.download_url(MODEL_URL, CACHE_ROOT, verbose=True)
            audeer.extract_archive(archive_path, MODEL_ROOT)
            print(f"Model extracted to: {MODEL_ROOT}")
        else:
            print("Model files found locally. Skipping download and extraction.")

        model = audonnx.load(MODEL_ROOT)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error during model loading: {e}")
        exit()


def get_audio_input(audio_file_path: str, target_sampling_rate: int) -> np.ndarray:
    """Loads, processes, and resamples an audio file."""
    print(f"\nLoading audio from: {audio_file_path}")
    try:
        data, orig_samplerate = sf.read(audio_file_path, dtype='float32')

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if orig_samplerate != target_sampling_rate:
            print(f"Resampling audio from {orig_samplerate} Hz to {target_sampling_rate} Hz...")
            gcd = np.gcd(target_sampling_rate, orig_samplerate)
            up = target_sampling_rate // gcd
            down = orig_samplerate // gcd
            signal = resample_poly(data, up, down).astype(np.float32)
        else:
            signal = data

        print(f"Audio loaded and processed. Signal shape: {signal.shape}")
        return signal

    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file_path}")
        exit()
    except Exception as e:
        print(f"Error processing audio file: {e}")
        exit()


def predict_age_gender(model: audonnx.Model, signal: np.ndarray, sampling_rate: int) -> tuple[float, str]:
    """Performs age and gender prediction."""
    print("\nPerforming prediction...")
    output = model(signal, sampling_rate)

    logits_age_array = output['logits_age']
    predicted_age_score = logits_age_array.item()
    actual_predicted_age = predicted_age_score * 100

    logits_gender_array = output['logits_gender']
    position_gender_logit = logits_gender_array.argmax(axis=1)[0]

    predicted_gender_label = "Unknown"
    if 0 <= position_gender_logit < len(GENDER_LABELS):
        predicted_gender_label = GENDER_LABELS[position_gender_logit]
    else:
        print(f"Warning: Gender index {position_gender_logit} out of bounds.")

    return actual_predicted_age, predicted_gender_label


def main():
    """Main function to run the age and gender prediction application."""
    print("Starting Age and Gender Prediction Application...\n")

    model = load_model()

    audio_file = 'Data/Input/sample_audio.wav'
    if not os.path.exists(audio_file):
        print(f"Creating a dummy audio file for testing: {audio_file}")
        dummy_signal = np.random.normal(size=TARGET_SAMPLING_RATE * 5).astype(np.float32)
        sf.write(audio_file, dummy_signal, TARGET_SAMPLING_RATE)
        print("Dummy audio file created.")

    signal_data = get_audio_input(audio_file, TARGET_SAMPLING_RATE)

    predicted_age, predicted_gender = predict_age_gender(model, signal_data, TARGET_SAMPLING_RATE)

    print("\n--- Final Prediction Results ---")
    print(f"Predicted Age: {predicted_age:.2f} years")
    print(f"Predicted Gender: {predicted_gender}")


if __name__ == "__main__":
    main()