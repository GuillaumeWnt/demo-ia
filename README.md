# Démo IA Locale — Chat LLM & Génération Musicale

Projet d'expérimentation d'IA en local sous Windows 11, optimisé pour tirer parti de l'accélération matérielle Nvidia CUDA (testé sur RTX 3070 Ti 8 Go VRAM) et géré avec l'outil de packaging moderne **`uv`**.

---

## Fonctionnalités

1. **Assistant de discussion textuelle (`main.py`)** :
   * Modèle : `Qwen/Qwen2.5-1.5B-Instruct` chargé en `bfloat16`.
   * Génération de texte en flux continu (*streaming*) dans le terminal.
   * Nettoyage du tampon clavier (`msvcrt`) pour éviter la validation intempestive d'entrées vides sur de longues requêtes sous Windows.

2. **Génération audio texte-vers-musique (`musique.py`)** :
   * Modèle : `facebook/musicgen-small` de Meta AI (300M de paramètres) chargé en `float16`.
   * Contrôle direct de la durée en secondes (50 tokens = 1 seconde d'audio).
   * Export automatique au format WAV stéréo/mono compatible (`float32`).

---

## Prérequis

* **Système d'exploitation** : Windows 11 (ou 10).
* **Matériel recommandé** : GPU Nvidia avec au moins 6 à 8 Go de VRAM (architecture Ampere ou supérieure pour le support natif du `bfloat16`).
* **Pilotes Nvidia** : Pilotes graphiques récents prenant en charge CUDA 12+.
* **Gestionnaire d'environnement** : [`uv`](https://docs.astral.sh/uv/) installé sur la machine.

---

## 1. Installation de `uv`

Vous pouvez installer `uv` de deux manières sous Windows :

### Méthode officielle (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Via Winget (Gestionnaire de paquets Windows)
```powershell
winget install --id=astral-sh.uv -e
```

> Après l'installation, redémarrez votre terminal pour que la variable d'environnement `PATH` soit prise en compte, puis vérifiez avec :
> ```powershell
> uv --version
> ```

---

## 2. Configuration du projet

Le fichier [`pyproject.toml`](./pyproject.toml) étant déjà versionné dans le dépôt Git, il contient déjà la configuration des dépendances et de l'index PyTorch CUDA 12.4 (`pytorch-cu124`).

Pour initialiser l'environnement virtuel et installer automatiquement toutes les dépendances compatibles GPU :

```powershell
# Cloner le dépôt et se placer à la racine (si ce n'est pas déjà fait)
cd demo-ia

# Créer le venv et installer toutes les dépendances
uv sync
```

---

## 3. Vérification de PyTorch & CUDA

Avant d'exécuter les modèles, vérifiez que PyTorch détecte bien votre carte graphique Nvidia :

```powershell
uv run python -c "import torch; print('CUDA disponible :', torch.cuda.is_available()); print('GPU détecté     :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun')"
```

Si tout est configuré correctement, `CUDA disponible : True` s'affichera ainsi que le nom de votre GPU (ex: `NVIDIA GeForce RTX 3070 Ti`).

---

## 4. Lancement des démonstrations

### Chat LLM interactif
Démarre la session de discussion avec Qwen 2.5 1.5B en streaming :
```powershell
uv run python main.py
```
*(Tapez `exit` ou `quit` pour quitter la session de discussion)*

### Génération musicale
Génère un fichier audio `musique.wav` à partir d'un prompt textuel avec MusicGen :
```powershell
uv run python musique.py
```

---

## 5. Surveillance de la VRAM en temps réel

Pour suivre la charge GPU, la température et l'occupation de la mémoire vidéo (VRAM) pendant les inférences, ouvrez un second terminal PowerShell et lancez :

```powershell
nvidia-smi -l 1
```

*(L'option `-l 1` actualise les statistiques chaque seconde. Appuyez sur `Ctrl + C` pour arrêter la surveillance).*