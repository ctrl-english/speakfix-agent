"""SpeakFix, fragment 1 : vérification de prononciation anglaise.

Le workflow reçoit une phrase cible et l'enregistrement de l'apprenant
(base64 ou URL), le transcrit avec Voxtral STT (dans une activité), puis
aligne mot à mot ce qui a été entendu avec la phrase cible. Il renvoie un
score, le statut de chaque mot et des conseils phonétiques ciblés pour
francophones (th, h aspiré, voyelles longues, -ed final, lettres muettes,
r anglais).

L'alignement et la construction du feedback sont des calculs purs : ils
tournent dans le code du workflow. Seul l'appel réseau (Voxtral) est dans
une activité, conforme aux règles de déterminisme du SDK.
"""

from __future__ import annotations

import asyncio
import base64

from pydantic import BaseModel, Field, model_validator

import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai import get_mistral_client

# ------------------------------------------------------------------ entrées / sorties


class PronunciationInput(BaseModel):
    target_phrase: str = Field(
        description="La phrase d'anglais que l'apprenant a essayé de dire."
    )
    audio_base64: str | None = Field(
        default=None,
        description="Enregistrement audio encodé en base64 (wav, mp3, ogg, webm, m4a...).",
    )
    audio_url: str | None = Field(
        default=None,
        description="URL publique de l'enregistrement (alternative à audio_base64).",
    )
    audio_file_name: str = Field(
        default="recording.wav",
        description="Nom de fichier avec son extension (aide la lecture de l'audio).",
    )

    @model_validator(mode="after")
    def _require_audio(self) -> "PronunciationInput":
        if not self.audio_base64 and not self.audio_url:
            raise ValueError(
                "Fournir audio_base64 ou audio_url (l'un des deux est obligatoire)."
            )
        return self


class WordResult(BaseModel):
    word: str = Field(description="Mot de la phrase cible (ou mot supplémentaire entendu).")
    status: str = Field(
        description="ok | close | sub | missing | extra",
    )
    heard: str | None = Field(
        default=None, description="Ce que Voxtral a entendu à la place du mot visé."
    )
    tip: str | None = Field(default=None, description="Conseil phonétique ciblé, en français.")


class PronunciationResult(BaseModel):
    target_phrase: str
    heard_phrase: str = Field(description="Transcription Voxtral de l'enregistrement.")
    score: int = Field(description="Score de prononciation sur 100.")
    words: list[WordResult]
    feedback: str = Field(description="Bilan et conseil principal, en français.")


# ------------------------------------------------------------------ conseils phonétiques (données)

WORD_TIPS: dict[str, str] = {
    "think": "θ : mets la pointe de la langue entre les dents et souffle. Attention, pas de « s » (sink) ni de « f ».",
    "this": "ð : même position de langue que θ, mais avec la voix qui vibre.",
    "third": "Souffle la langue entre les dents sur ce mot : pas de « s » au départ.",
    "month": "Termine avec la langue visible entre les dents : le mot finit sur un souffle, pas sur un « s ».",
    "mother": "le th du milieu est voisé (ð) : les cordes vocales vibrent.",
    "father": "le th du milieu est voisé (ð) : les cordes vocales vibrent.",
    "together": "le th du milieu est doux et voisé (ð).",
    "something": "le mot se termine par θ, la langue reste entre les dents.",
    "nothing": "le mot se termine par θ, pas par un « s ».",
    "than": "ce mot contient un th, pas un « d ».",
    "he": "expires un petit coup sur le h de départ, il s'entend.",
    "has": "le h est audible : pas « as », bien « has ».",
    "have": "le h est audible : pas « ave », bien « have ».",
    "hard": "souffle bien le h du départ.",
    "hat": "souffle le h du départ.",
    "head": "souffle le h du départ.",
    "how": "le h se souffle avant le son « aou ».",
    "high": "respiration audible sur le h, puis la voyelle se prolonge.",
    "hill": "h aspiré du départ, voyelle brève.",
    "house": "h bien expiré du départ : on l'entend comme un léger souffle.",
    "her": "h aspiré du départ.",
    "husband": "h aspiré du départ.",
    "heavy": "h aspiré du départ.",
    "behind": "le h du milieu est aussi aspiré.",
    "sheep": "voyelle longue iː : étire les lèvres en sourire, le mot dure.",
    "ship": "voyelle brève ɪ : bouche détendue, très court.",
    "sleeping": "la première voyelle est longue (iː), la seconde est brève.",
    "still": "voyelle brève ɪ, bouche relâchée.",
    "sees": "voyelle longue iː, lèvres étirées.",
    "seal": "voyelle longue iː.",
    "beans": "voyelle longue iː.",
    "need": "voyelle longue iː.",
    "beach": "voyelle longue iː.",
    "these": "voyelle longue iː et th voisé ð.",
    "heat": "voyelle longue iː.",
    "watched": "le -ed se prononce /t/ : « watched », pas « watch-ed ».",
    "played": "le -ed se prononce /d/ : une seule syllabe.",
    "decided": "après un d, le -ed fait /ɪd/ : de-ci-ded, trois syllabes.",
    "wanted": "après un t, le -ed fait /ɪd/ : wan-ted.",
    "needed": "le -ed fait /ɪd/ : ne-ded.",
    "finished": "le -ed fait /t/ : une seule syllabe à la fin.",
    "relaxed": "le -ed fait /t/ : le mot se termine de façon sèche.",
    "walk": "le l est muet : « wok ».",
    "knife": "le k est muet : on prononce « naïf ».",
    "castle": "le t est muet.",
    "subtle": "le b est muet.",
    "calm": "le l est muet.",
    "island": "le s est muet : « aï-land ».",
    "wrong": "le w est muet.",
    "knows": "le k est muet.",
    "listen": "le t est muet.",
    "answer": "le w est muet.",
    "christmas": "le t est muet.",
    "red": "r doux du départ, la langue ne touche pas le palais.",
    "river": "r doux, pas de r roulé.",
    "very": "r doux à la fin, très léger.",
    "rural": "commence par un r doux, suivi du son « ou ».",
    "really": "voyelle longue iː puis r léger.",
    "prefers": "r final léger, non roulé.",
    "bread": "le r du départ est doux, la voyelle est large.",
}

FIXED_PHRASE_RULES: list[str] = [
    "Le son TH : langue entre les dents, sans voix (θ, think) ou avec la voix (ð, this).",
    "Le H aspiré : une petite expiration audible au départ des mots comme he, house, have.",
    "Voyelles : ɪ (ship) est court et détendu, iː (sheep) est long, lèvres étirées.",
    "Le -ED final : /t/ après un son sourd (watched), /d/ après un son sonore (played), /ɪd/ après t ou d (decided).",
    "Lettres muettes : attention à knife, castle, island, calm, listen.",
    "Le R anglais : tout doux, la langue ne touche pas le palais, jamais roulé.",
]

# ------------------------------------------------------------------ calculs purs (autorisés dans le workflow)


def _norm(word: str) -> str:
    """Normalise un mot : minuscules, sans ponctuation, apostrophes collées."""
    s = word.lower()
    s = "".join(ch for ch in s if ch.isalpha() or ch == "'")
    s = s.replace("'", "")
    variants = {"gonna": "going", "wanna": "want", "cuz": "because", "yeah": "yes"}
    return variants.get(s, s)


def _lev(a: str, b: str) -> int:
    """Distance de Levenshtein entre deux mots normalisés."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _sub_cost(target: str, heard: str) -> float:
    """Coût de substitution : 0 identique, 0.5 quasi-identique (variante de transcription), 1 sinon."""
    if target == heard:
        return 0.0
    if len(target) >= 4 and len(heard) >= 4 and _lev(target, heard) <= 1:
        return 0.5
    return 1.0


def align_words(target_phrase: str, heard_phrase: str) -> list[tuple[str, str, str | None]]:
    """Aligne la phrase cible et la phrase entendue, mot à mot.

    Renvoie une liste de tuples (statut, mot_visé, mot_entendu) :
    statut parmi "ok", "close", "sub", "missing", "extra".
    """
    targets = [_norm(w) for w in target_phrase.split() if _norm(w)]
    heard = [_norm(w) for w in heard_phrase.split() if _norm(w)]
    m, n = len(targets), len(heard)

    # Alignement par programmation dynamique (édition Levenshtein sur les mots).
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + _sub_cost(targets[i - 1], heard[j - 1]),
            )

    ops: list[tuple[str, str, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + _sub_cost(targets[i - 1], heard[j - 1]):
            t, h = targets[i - 1], heard[j - 1]
            if t == h:
                ops.append(("ok", t, None))
            elif _sub_cost(t, h) <= 0.5:
                ops.append(("close", t, h))
            else:
                ops.append(("sub", t, h))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1.0:
            ops.append(("missing", targets[i - 1], None))
            i -= 1
        else:
            ops.append(("extra", heard[j - 1], None))
            j -= 1
    ops.reverse()
    return ops


def _tip_for(word: str, status: str, heard: str | None) -> str | None:
    """Choisit le conseil phonétique le plus pertinent pour un mot raté."""
    w = word.lower()
    known = WORD_TIPS.get(w)
    if status == "sub" and heard:
        if known:
            return f"J'ai entendu « {heard} » au lieu de « {word} » : {known}"
        if "th" in w:
            return f"J'ai entendu « {heard} » au lieu de « {word} » : le th se dit langue entre les dents."
        if w.startswith("h") and heard == w[1:]:
            return "Le h a disparu : en anglais il s'expire, on doit l'entendre."
        if w.endswith("ed"):
            return f"J'ai entendu « {heard} » : vérifie le son du -ed final (/t/, /d/ ou /ɪd/)."
        return f"J'ai entendu « {heard} » au lieu de « {word} ». Réécoute le modèle et répète lentement."
    if status == "missing":
        if known:
            return known
        return f"Le mot « {word} » n'a pas été entendu. Dis-le seul, puis refais la phrase."
    return known


def build_feedback(score: int, words: list[WordResult]) -> str:
    """Bilan global en français selon le score et les erreurs détectées."""
    if score >= 80:
        return "Superbe prononciation, le son est bien installé. Continue avec des phrases plus longues."
    if score >= 50:
        return "Bonne base. Concentre-toi sur les mots en rouge : écoute le modèle, puis répète lentement chaque son difficile avant de refaire la phrase entière."
    return "Écoute d'abord le modèle en entier, puis mot par mot les mots en rouge, et répète. Refais ensuite la phrase entière à voix haute."


# ------------------------------------------------------------------ activité (appels réseau)


@workflows.activity()
async def transcribe_recording(
    audio_base64: str | None, audio_url: str | None, audio_file_name: str
) -> str:
    """Transcrit l'enregistrement avec Voxtral STT et renvoie le texte entendu."""
    # Récupération des octets : décodage base64 ou téléchargement de l'URL.
    if audio_base64:
        audio_bytes = base64.b64decode(audio_base64)
        file_name = audio_file_name
    else:
        import urllib.request

        with urllib.request.urlopen(audio_url, timeout=60) as response:  # noqa: S310
            audio_bytes = response.read()
        file_name = audio_url.rsplit("/", 1)[-1] or audio_file_name

    client = get_mistral_client()

    def _transcribe() -> str:
        transcription = client.audio.transcriptions.complete(
            model="voxtral-mini-transcribe-3",
            file={"file_name": file_name, "content": audio_bytes},
            language="en",
        )
        return (transcription.text or "").strip()

    # L'appel SDK est bloquant : on l'exécute dans un thread pour ne pas
    # bloquer la boucle d'événements du worker.
    return await asyncio.to_thread(_transcribe)


# ------------------------------------------------------------------ workflow


@workflows.workflow.define(
    name="pronunciation-check",
    workflow_display_name="SpeakFix : check de prononciation",
    workflow_description=(
        "Transcrit un enregistrement de l'apprenant avec Voxtral STT, l'aligne mot à mot "
        "avec la phrase cible et renvoie score, mots ratés et conseils phonétiques."
    ),
)
class PronunciationCheckWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: PronunciationInput) -> PronunciationResult:
        # 1. Transcription (réseau, dans une activité).
        heard = await transcribe_recording(
            input.audio_base64, input.audio_url, input.audio_file_name
        )

        # 2. Alignement et score (calculs purs, dans le workflow).
        ops = align_words(input.target_phrase, heard)
        words: list[WordResult] = []
        ok_count = 0
        close_count = 0
        target_count = sum(1 for status, _, _ in ops if status != "extra")

        for status, word, heard_word in ops:
            if status == "ok":
                ok_count += 1
                words.append(WordResult(word=word, status="ok"))
            elif status == "close":
                close_count += 1
                words.append(WordResult(word=word, status="close", heard=heard_word))
            elif status == "sub":
                words.append(WordResult(word=word, status="sub", heard=heard_word))
            elif status == "missing":
                words.append(WordResult(word=word, status="missing"))
            else:
                words.append(WordResult(word=word, status="extra"))

        # Conseils phonétiques pour les mots problématiques.
        for item in words:
            if item.status in ("sub", "missing", "close"):
                item.tip = _tip_for(item.word, item.status, item.heard)
        tipped = [w for w in words if w.tip]

        score = 0
        if target_count:
            score = round((ok_count + close_count * 0.75) * 100 / target_count)

        # 3. Bilan global + rappel de la règle de la série si nécessaire.
        feedback = build_feedback(score, words)
        if score < 60 and tipped:
            feedback += " " + FIXED_PHRASE_RULES[0]

        return PronunciationResult(
            target_phrase=input.target_phrase,
            heard_phrase=heard,
            score=score,
            words=words,
            feedback=feedback,
        )
