from typing import Any

import scipy.io.wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

device = "cuda"
model_id = "facebook/musicgen-small"

print("1. Chargement de MusicGen sur la RTX 3070 Ti...")
processor = AutoProcessor.from_pretrained(model_id)
model: Any = MusicgenForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",  # Place automatiquement le modèle sur la RTX 3070 Ti
)

description = "rain sound effects with lightning, distant thunder, dark and dramatic ambient atmosphere"
print(f"2. Prompt : « {description} »...")

# Les entrées doivent être envoyées sur le même périphérique (CUDA)
inputs = processor(
    text=[description],
    padding=True,
    return_tensors="pt",
).to(device)

duree_secondes = 10
tokens_a_generer = int(duree_secondes * 50)

print(f"3. Génération de {duree_secondes}s de son...")
with torch.no_grad():
    audio_values = model.generate(
        **inputs,
        max_new_tokens=tokens_a_generer,
        do_sample=True,
    )

# Conversion vers le CPU avant l'écriture sur disque
scipy.io.wavfile.write(
    "musique.wav",
    rate=model.config.audio_encoder.sampling_rate,
    data=audio_values[0, 0].to(torch.float32).cpu().numpy(),
)

print("Terminé ! Fichier généré dans 'musique.wav'")
