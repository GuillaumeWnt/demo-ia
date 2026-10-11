from typing import Any

import scipy.io.wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

model_id = "facebook/musicgen-small"

print("1. Chargement du processeur et du modèle...")
processor = AutoProcessor.from_pretrained(model_id)
model: Any = MusicgenForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.float32,  # ou torch.bfloat16
)

description = "rain sound effects with lightning, distant thunder, dark and dramatic ambient atmosphere"
print(f"2. Préparation du prompt : « {description} »...")

inputs = processor(
    text=[description],
    padding=True,
    return_tensors="pt",
)

# 256 tokens audio correspondent environ à 5 secondes de musique
print("3. Génération de la musique en cours...")
with torch.no_grad():
    audio_values = model.generate(
        **inputs,
        max_new_tokens=500,  # environ 10 secondes de musique
        do_sample=True,
    )

# 4. Sauvegarde dans un fichier WAV
sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write(
    "musique.wav",
    rate=sampling_rate,
    data=audio_values[0, 0].numpy(),
)

print("Musique générée avec succès dans 'musique.wav' !")
