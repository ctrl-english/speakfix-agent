# SpeakFix Agent

Agent de correction de prononciation anglaise pour les élèves, construit sur les **Mistral Workflows** (Studio).

## Fragment 1 : `pronunciation-check`

Le premier fragment de technologie (un à la fois !) :

1. L'apprenant enregistre sa voix en disant une phrase d'anglais.
2. Le workflow transcrit l'audio avec **Voxtral STT** (`voxtral-mini-transcribe-3`).
3. Il aligne mot à mot la transcription avec la phrase cible (distance de Levenshtein pondérée).
4. Il renvoie un **score sur 100**, le **statut de chaque mot** (ok / close / sub / missing / extra) et des **conseils phonétiques en français** ciblant les erreurs typiques des francophones (th, h aspiré, voyelles longues ɪ/iː, -ed final, lettres muettes, r anglais).

### Entrée

```json
{
  "target_phrase": "I think this is the third month.",
  "audio_url": "https://exemple.com/enregistrement.mp3",
  "audio_file_name": "recording.mp3"
}
```

`audio_base64` (audio encodé en base64) est accepté à la place de `audio_url`. Un seul des deux est obligatoire.

### Sortie

```json
{
  "target_phrase": "I think this is the third month.",
  "heard_phrase": "I sink this is the third month",
  "score": 86,
  "words": [
    { "word": "i", "status": "ok" },
    { "word": "think", "status": "sub", "heard": "sink",
      "tip": "J'ai entendu « sink » au lieu de « think » : θ : mets la pointe de la langue entre les dents et souffle..." }
  ],
  "feedback": "Bonne base. Concentre-toi sur les mots en rouge..."
}
```

### Sécurité et données

- Aucune clé API ne figure dans ce dépôt : `.env` (clé personnelle pour le développement local) est exclu par `.gitignore`.
- Le worker déployé s'authentifie avec l'identité de service du déploiement provisionnée par la plateforme Mistral, pas avec une clé personnelle.
- Les enregistrements transitent uniquement vers l'API Mistral (hébergée dans l'Union européenne) pour la transcription.

### Déploiement

Worker Python (`apps/worker`) déployé en workflow managé Mistral depuis ce dépôt.

- Build : `deploy/docker/Dockerfile.worker` (résout les dépendances avec `uv` en build public PyPI).
- Workflow : `apps/worker/src/worker/workflows/pronunciation.py`.
- Tests purs de l'alignement : `apps/worker/tests/test_pronunciation_alignment.py` (exécutables sans le SDK : `python3 tests/test_pronunciation_alignment.py`).

### Roadmap (fragments suivants)

- Fragment 2 : audio modèle généré par Voxtral TTS dans la réponse.
- Fragment 3 : détection automatique des sons à retravailler sur corpus libre (sans phrase cible).
- Fragment 4 : interface web avec micro (App Studio) et suivi individualisé.
