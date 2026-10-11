from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

print("1. Téléchargement et chargement du Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=False,  # Refuse explicitement tout code tiers
)

print("2. Téléchargement et chargement des poids du modèle (.safetensors)...")
model: Any = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,  # Refuse explicitement tout code tiers
    dtype=torch.float32,  # Calcul standard sur processeur (CPU)
)

# TextStreamer affiche les mots calculés au fur et à mesure
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
# Initialisation de la discussion avec la consigne système en premier élément
messages: list[dict[str, str]] = []

print("\n--- Session de chat démarrée (tape 'exit' pour quitter) ---\n")

while True:
    user_input = input("Toi : ")
    if user_input.strip().lower() in ["exit", "quit", "q"]:
        break
    if not user_input.strip():
        continue

    # 3. Préparer le message au format chat et enrichir l'historique
    messages.append({"role": "user", "content": user_input})

    # Convertir la discussion selon le template attendu par le modèle
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    print("Modèle : ", end="", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    # 4. Décoder uniquement les nouveaux tokens générés pour conserver l'historique
    reponse = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    messages.append({"role": "assistant", "content": reponse})
    print()
