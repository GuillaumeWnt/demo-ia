import msvcrt
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("1. Téléchargement et chargement du Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

print("2. Chargement du modèle sur la RTX 3070 Ti...")
model: Any = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,
    dtype=torch.bfloat16,  # Format idéal pour l'architecture Ampere de la 3070 Ti
    device_map="cuda",  # Charge et alloue automatiquement le modèle sur le GPU
)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
messages: list[dict[str, str]] = []


def vider_buffer_clavier() -> None:
    """Vide toutes les frappes en attente dans la console Windows."""
    while msvcrt.kbhit():
        msvcrt.getwch()


print("\n--- Session de chat démarrée (tape 'exit' pour quitter) ---\n")

while True:
    # 1. On nettoie tout ce qui a pu être tapé pendant le calcul
    vider_buffer_clavier()

    user_input = input("Toi : ")

    # 2. Gestion de la sortie
    if user_input.strip().lower() in ["exit", "quit", "q"]:
        break

    # 3. Sécurité : si l'entrée est vide (ex: simple appui sur Entrée), on redemande
    if not user_input.strip():
        continue

    messages.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Modèle : ", end="", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    reponse = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    messages.append({"role": "assistant", "content": reponse})
    print()
