import audeer
import audonnx
import numpy as np


#Settings part
url = 'https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'
cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')
sampling_rate = 16000

archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root)

signal = np.random.normal(size=sampling_rate).astype(np.float32)
output = model(signal, sampling_rate)
print(output)#to check prediction result

# For logits_age
logits_age_array = output['logits_age']
position_age_logit = logits_age_array.argmax()
# For a (1,1) array, the actual value is usually what's of interest, or it's implicitly index 0
actual_age_score = logits_age_array.item()
actual_predicted_age = actual_age_score * 100

# For logits_gender
logits_gender_array = output['logits_gender']
position_gender_logit = logits_gender_array.argmax(axis=1)[0]


print("\nExtracted Positions (Indices) of Highest Values:")
print(f"Highest logits_age value: {actual_predicted_age}")

gender_labels = ['female', 'male', 'children']
if 0 <= position_gender_logit < len(gender_labels):
    predicted_gender = gender_labels[position_gender_logit]
    print(f"Predicted Gender: {predicted_gender}")
else:
    print(f"Gender index {position_gender_logit} is out of bounds for defined labels.")

print(f"Position (index) of highest logits_age value: {position_age_logit}")
print(f"Position (index) of highest logits_gender value: {position_gender_logit}")
