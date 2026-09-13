"""
Tracker de sujets déjà utilisés, pour empêcher le pipeline de régénérer
plusieurs vidéos sur le même sujet réel (ex: 5 videos sur le Palais des
Papes d'Avignon avec des titres reformulés différemment) ET pour empêcher
la sur-répétition d'un même CONCEPT récurrent (ex: "tunnel secret") même
quand le lieu précis change à chaque fois.

CORRECTIF APPLIQUÉ (voir historique) :
La version précédente mettait "tunnel", "souterrain", "château", "palais",
"secret", "cache" etc. dans STOPWORDS. Ces mots étaient donc retirés des
deux titres AVANT le calcul de similarité, ce qui rendait impossible la
détection de doublons thématiques du type "Le tunnel secret de la
Conciergerie" vs "Le tunnel oublié de la Bastille" (score mesuré : 0.40,
sous le seuil de 0.6, car seuls les noms de lieux différents restaient
comparés). Résultat concret observé en production : ~25 sujets sur 65
contenaient le mot "tunnel" sans jamais être bloqués par is_duplicate_topic.

Deux corrections independantes et complementaires :
1. STOPWORDS ne contient plus que les mots vraiment vides de sens
   (articles, prepositions, mots meta comme "mystere"/"secret"/"decouverte").
   Les mots de TYPE DE LIEU/CONCEPT (tunnel, chateau, souterrain...) restent
   dans le calcul de similarite classique.
2. Un second garde-fou INDEPENDANT du nom de lieu : CONCEPT_GROUPS regroupe
   les synonymes d'un meme concept recurrent (ex: tunnel/souterrain/passage
   secret/catacombe = meme "famille"). Si un concept de cette famille a ete
   utilise trop souvent recemment (>= CONCEPT_MAX_RECENT_USES sur les
   CONCEPT_LOOKBACK derniers sujets), le nouveau sujet est aussi considere
   comme un doublon, meme si le lieu precis est different.

Utilisation typique dans brain.py (inchangee) :

    from modules.utils.topic_tracker import (
        load_topic_history, is_duplicate_topic, record_topic_usage,
    )

    used_topics = load_topic_history()
    for attempt in range(MAX_RETRIES):
        candidate = <generation LLM du sujet>
        is_dup, matched = is_duplicate_topic(candidate, used_topics)
        if not is_dup:
            record_topic_usage(candidate)
            return candidate
        # sinon on regenere
"""

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher

TOPIC_HISTORY_PATH = os.path.join(
    os.getcwd(), "assets", "state", "topic_history.json"
)

# Seuil de similarité (0-1) au-dessus duquel deux sujets sont considérés
# comme le même sujet réel malgré une reformulation différente.
DEFAULT_SIMILARITY_THRESHOLD = 0.6

# Nombre max de sujets conservés dans l'historique (evite un fichier qui
# grossit indefiniment ; les plus anciens sont les moins pertinents pour
# la dedup, un pipeline qui tourne depuis des mois ne devrait pas comparer
# un nouveau sujet a une video vieille de 2 ans).
MAX_HISTORY_SIZE = 500

# --------------------------------------------------------------------
# STOPWORDS : uniquement des mots réellement vides de sens (articles,
# prépositions, mots "méta" génériques). On NE met plus ici les mots qui
# désignent un TYPE de lieu ou de concept (tunnel, château, secret...),
# car ce sont justement ceux qui permettent de détecter la sur-répétition
# d'un même type de sujet.
# --------------------------------------------------------------------
STOPWORDS = {
    "le", "la", "les", "l", "de", "des", "du", "un", "une", "et", "en",
    "sur", "sous", "dans", "au", "aux", "a", "d",
    "revele", "revelee", "decouvert", "decouverte",
    "apres", "ans", "oubli",
    "france", "francaise", "francais",
    "cite", "ville",
    "trouve", "trouvee",
    "histoire", "historique", "veritable", "reelle", "reel",
}

# --------------------------------------------------------------------
# CONCEPT_GROUPS : familles de mots-clés désignant le MÊME concept
# récurrent, indépendamment du lieu précis mentionné. Sert de garde-fou
# supplémentaire pour éviter que "tunnel de la Conciergerie" puis "tunnel
# de la Bastille" puis "passage secret de Notre-Dame" passent tous comme
# des sujets "différents" simplement parce que le lieu change.
# --------------------------------------------------------------------
CONCEPT_GROUPS = {
    "souterrain": {
        "tunnel", "tunnels", "souterrain", "souterrains", "souterraine",
        "passage", "catacombe", "catacombes", "galerie", "galeries",
        "cave", "caves", "crypte", "cryptes",
    },
    "salle_cachee": {
        "salle", "chambre", "piece", "cachee", "cache", "caches",
        "secrete", "secretes",
    },
    "tresor": {
        "tresor", "tresors", "or", "bijou", "bijoux", "magot",
    },
    "malediction": {
        "malediction", "maudit", "maudite", "sort", "sortilege",
    },
}

# Fenêtre glissante sur laquelle on évalue la sur-répétition d'un concept.
CONCEPT_LOOKBACK = 8
# Nombre max d'usages tolérés d'un même concept dans cette fenêtre avant
# de considérer tout nouveau sujet du même concept comme un doublon.
CONCEPT_MAX_RECENT_USES = 2


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _keywords(title: str) -> set:
    text = _strip_accents(title.lower())
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    words = re.split(r"[\s\-]+", text)
    return {w for w in words if w and w not in STOPWORDS and len(w) > 2}


def _similarity(title_a: str, title_b: str) -> float:
    kw_a, kw_b = _keywords(title_a), _keywords(title_b)
    if not kw_a or not kw_b:
        overlap = 0.0
    else:
        overlap = len(kw_a & kw_b) / min(len(kw_a), len(kw_b))
    seq = SequenceMatcher(None, title_a.lower(), title_b.lower()).ratio()
    return max(overlap, seq * 0.7)


def _concepts_in_title(title: str) -> set:
    """Retourne l'ensemble des familles conceptuelles présentes dans le titre."""
    kw = _strip_accents(title.lower())
    kw = re.sub(r"[^a-z0-9\s\-]", " ", kw)
    words = set(re.split(r"[\s\-]+", kw))
    found = set()
    for concept_name, variants in CONCEPT_GROUPS.items():
        if words & variants:
            found.add(concept_name)
    return found


def _is_concept_overused(candidate: str, recent_history: list) -> str | None:
    """
    Vérifie si le candidat appartient à un concept déjà trop utilisé dans
    les derniers sujets. Retourne le nom du concept sur-utilisé, ou None.
    """
    candidate_concepts = _concepts_in_title(candidate)
    if not candidate_concepts:
        return None

    window = recent_history[-CONCEPT_LOOKBACK:] if recent_history else []
    for concept in candidate_concepts:
        count = sum(1 for past in window if concept in _concepts_in_title(past))
        if count >= CONCEPT_MAX_RECENT_USES:
            return concept
    return None


def load_topic_history():
    """Charge la liste des sujets déjà utilisés (liste de strings)."""
    if not os.path.exists(TOPIC_HISTORY_PATH):
        return []
    try:
        with open(TOPIC_HISTORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(t) for t in data]
        return []
    except Exception as e:
        print(f"⚠️ Impossible de charger topic_history.json, on repart de zéro : {e}")
        return []


def is_duplicate_topic(candidate, history, threshold=DEFAULT_SIMILARITY_THRESHOLD):
    """
    Retourne (True, raison) si le candidat :
    - ressemble trop (similarité textuelle/mots-clés) à un sujet déjà
      utilisé, OU
    - appartient à une famille conceptuelle (tunnel/souterrain, salle
      cachée, trésor, malédiction...) sur-utilisée récemment, même avec
      un lieu différent à chaque fois.
    Sinon retourne (False, None).
    """
    if not candidate:
        return False, None

    for past_topic in history:
        sim = _similarity(candidate, past_topic)
        if sim >= threshold:
            return True, past_topic

    overused_concept = _is_concept_overused(candidate, history)
    if overused_concept:
        return True, f"concept sur-utilisé récemment : '{overused_concept}'"

    return False, None


def record_topic_usage(topic):
    """Ajoute un sujet à l'historique persistant et le sauvegarde."""
    if not topic:
        return
    history = load_topic_history()
    history.append(topic)
    if len(history) > MAX_HISTORY_SIZE:
        history = history[-MAX_HISTORY_SIZE:]

    os.makedirs(os.path.dirname(TOPIC_HISTORY_PATH), exist_ok=True)
    try:
        with open(TOPIC_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Impossible d'écrire topic_history.json : {e}")
