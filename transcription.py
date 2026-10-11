import os
from typing import Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Utilisation de la version large-v3-turbo : 4x plus rapide, ~2 Go de VRAM et qualité équivalente
model_id = "openai/whisper-large-v3-turbo"
device = "cuda:0"
dtype = torch.float16

print("1. Chargement de Whisper Large-v3-Turbo sur la RTX 3070 Ti...")

# Chargement du modèle directement dans la VRAM en float16
model: Any = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# 2. Configuration du pipeline de reconnaissance vocale
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,  # Découpe en fenêtres de 30s pour gérer les fichiers de toute durée
    return_timestamps=True,  # Génère les sous-titres avec minutage
)

# 3. Fichier audio à transcrire (remplacez par un fichier contenant de la voix)
fichier_audio = "sample3_tresLong.mp3"

if not os.path.exists(fichier_audio):
    raise FileNotFoundError(f"Le fichier '{fichier_audio}' est introuvable.")

print(f"2. Transcription du fichier '{fichier_audio}' en cours...")
resultat = pipe(
    fichier_audio,
    generate_kwargs={
        # "language": "french",  # Force le français (ou retirez pour détection automatique)
        "task": "transcribe",  # "transcribe" pour transcrire, "translate" pour traduire en anglais
    },
)

# Sécurisation du typage pour le linter (le pipeline retourne un dictionnaire pour un fichier unique)
assert isinstance(resultat, dict)

# 4. Affichage du résultat
print("\n--- Transcription complète ---")
print(resultat.get("text", ""))

print("\n--- Segments avec minutage ---")
chunks: list[dict[str, Any]] = resultat.get("chunks", [])
for chunk in chunks:
    debut, fin = chunk["timestamp"]
    fin_str = f"{fin:05.2f}s" if fin is not None else "   ?   "
    texte = chunk["text"]
    print(f"[{debut:05.2f}s -> {fin_str}] {texte}")
