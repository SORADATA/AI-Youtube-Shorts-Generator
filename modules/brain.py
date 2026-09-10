import os
import re
import json
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv
from constants import (
    GROQ_MODEL,
    OPENROUTER_FALLBACK_MODEL_1,
    OPENROUTER_FALLBACK_MODEL_2
)
from modules.utils.topic_tracker import load_topic_history, is_duplicate_topic

try:
    from modules.utils.client_http.zernio_client import get_latest_videos_stats
except ImportError:
    print("⚠️ Module zernio_client introuvable. Création de données factices pour le test.")
    def get_latest_videos_stats():
        return None

try:
    from modules.utils.wikipedia_grounding import fetch_grounding_source
    GROUNDING_AVAILABLE = True
except ImportError:
    GROUNDING_AVAILABLE = False
    def fetch_grounding_source(query, hint_country=None):
        return None

load_dotenv()

ACCENTED_CHARS = "éèêëàâäùûüçîïôœ"

ACCENT_INSTRUCTION = (
    "IMPERATIF ORTHOGRAPHE : le francais doit etre parfaitement accentue "
    "(é, è, ê, à, ù, ç, ô, î etc)."
)

NO_META_AI_INSTRUCTION = (
    "INTERDICTION ABSOLUE : ne jamais mentionner l'intelligence artificielle, "
    "l'IA, un algorithme, une technologie de generation de contenu, ou tout "
    "aspect meta concernant la creation de la video elle-meme. Le sujet et le "
    "texte doivent parler uniquement du mystere/de l'histoire reelle, jamais "
    "de l'outil ou de la methode utilisee pour le raconter."
)

VERACITY_INSTRUCTION = (
    "EXIGENCE DE VERACITE HISTORIQUE : "
    "1) N'utilise QUE des lieux, evenements et personnages REELS et documentes. "
    "Interdiction absolue d'inventer un nom de lieu, un evenement ou un personnage. "
    "2) Ne deplace JAMAIS geographiquement un fait reel. Si un evenement/lieu "
    "s'est produit dans un pays ou une region precise, cette localisation doit "
    "etre respectee EXACTEMENT dans le texte, le titre et le sujet. Ne jamais "
    "presenter un fait etranger comme francais (ou inversement) meme si l'histoire "
    "est fascinante. "
    "3) Le sujet doit rester peu connu du grand public mais reel et verifiable, "
    "jamais une invention presentee comme un fait. "
    "4) EVITE les noms de lieux trop generiques/ambigus (ex: une simple 'Eglise "
    "Saint-Pierre' sans precision) qui existent en plusieurs exemplaires dans "
    "differents pays -- precise toujours le nom complet et specifique du lieu "
    "(ex: 'Eglise Saint-Pierre d'Oron' plutot que juste 'Eglise Saint-Pierre'). "
    "5) Pour chaque lieu mentionne dans le champ 'location_name', precise aussi "
    "le pays reel dans un champ 'location_country' (ex: 'France', 'Allemagne', "
    "'Suisse'). Si le sujet annonce une zone geographique specifique, TOUS les "
    "lieux du script doivent appartenir a cette zone reelle. "
    "6) ATTENTION AUX LIEUX AMBIGUS AVEC MEME NOM : il existe souvent plusieurs "
    "lieux similaires (ex: plusieurs 'ponts du diable' en France, plusieurs "
    "'eglises Saint-Pierre') dans des villes/regions differentes. Verifie "
    "mentalement que le lieu precis correspond exactement au sujet donne, sans "
    "confondre deux legendes/lieux distincts portant un nom proche."
)


def _has_missing_accents(text, min_hits=3):
    suspicious_patterns = [
        r"\bdecouv", r"\bmyster", r"\bsecret", r"\bexplor",
        r"\btheori", r"\bphenomen", r"\bhistoi", r"\bevenem",
        r"\bepoque", r"\betrang", r"\brevel", r"\bdifferen",
        r"\ba ete\b", r"\bpeut etre\b", r"\binteresse",
    ]
    text_lower = text.lower()
    hits = sum(1 for p in suspicious_patterns if re.search(p, text_lower))
    has_any_accent = any(c in text_lower for c in ACCENTED_CHARS)
    return hits >= min_hits and not has_any_accent


AI_MENTION_PATTERNS = [
    r"intelligence\s+artificielle", r"\bIA\b", r"\bl'IA\b",
    r"artificial\s+intelligence", r"\bl'algorithme\b", r"\bchatgpt\b",
    r"\bgroq\b", r"\bgemini\b", r"genere[e]?\s+par\s+l'?ia",
]


def _contains_ai_mention(text):
    if not text:
        return False
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in AI_MENTION_PATTERNS)


def _format_stats_instruction(previous_stats_list, label="hooks"):
    if not previous_stats_list:
        return ""

    stats_text = "\n".join([
        f'- Titre : "{s.get("title", "?")}" | Vues : {s.get("views", "?")} | Likes : {s.get("likes", "?")}'
        for s in previous_stats_list
        if isinstance(s, dict) and "title" in s
    ])

    return f"""
ANALYSE DES PERFORMANCES RECENTES :
Voici les resultats de nos dernieres videos publiees :
{stats_text}

INSTRUCTION :
Adapte le {label} selon les performances sans citer les stats explicitement.
"""


def _clean_single_line_title(text):
    if not text:
        return ""
    cleaned = text.replace('"', '').replace('"', '').replace('"', '').strip()
    lines = [line.strip(' -•\t') for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    return re.sub(r"\s+", " ", lines[0]).strip()


def _clean_json_response(content):
    if not content:
        return content
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


FRENCH_GEO_KEYWORDS = [
    "france", "francaise", "francais", "bretagne", "normandie", "vendee",
    "charente", "gironde", "aquitaine", "atlantique francaise",
    "cote atlantique francaise", "cote d'azur", "provence", "occitanie",
]

COUNTRY_KEYWORDS = {
    "france": ["france", "francaise", "francais", "bretagne", "normandie",
               "vendee", "charente", "gironde", "aquitaine", "provence", "occitanie"],
    "suisse": ["suisse", "helvetique", "vaud", "geneve", "valais", "zurich", "berne"],
    "allemagne": ["allemagne", "allemand", "allemande", "baviere", "bavaria"],
    "italie": ["italie", "italien", "italienne", "toscane", "sicile"],
    "espagne": ["espagne", "espagnol", "espagnole", "catalogne", "andalousie"],
    "belgique": ["belgique", "belge"],
}


def _topic_claims_french_location(topic):
    topic_lower = topic.lower()
    return any(kw in topic_lower for kw in FRENCH_GEO_KEYWORDS)


def _guess_country_hint(topic):
    topic_lower = topic.lower()
    for country, keywords in COUNTRY_KEYWORDS.items():
        if any(kw in topic_lower for kw in keywords):
            return country.capitalize()
    return None


class WikidataChecker:
    API_URL = "https://www.wikidata.org/w/api.php"
    HEADERS = {
        "User-Agent": os.getenv(
            "WIKIMEDIA_CONTACT",
            "AuraContentPipeline/1.0 (contact non configure)"
        )
    }
    CACHE = {}
    MIN_DELAY_BETWEEN_CALLS = 0.6

    @classmethod
    def _throttled_get(cls, params, max_retries=2):
        for attempt in range(max_retries + 1):
            time.sleep(cls.MIN_DELAY_BETWEEN_CALLS)
            try:
                r = requests.get(cls.API_URL, params=params, headers=cls.HEADERS, timeout=10)
                if r.status_code == 429:
                    wait = 3 * (attempt + 1)
                    print(f"⏳ Wikidata 429, attente {wait}s avant retry...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    print(f"⚠️ Wikidata erreur reseau (abandon apres {max_retries + 1} tentatives) : {e}")
                    return None
                time.sleep(2)
        return None

    @classmethod
    def _search_entity_ids(cls, location_name, limit=3):
        params = {
            "action": "wbsearchentities",
            "search": location_name,
            "language": "fr",
            "format": "json",
            "limit": limit,
            "type": "item",
        }
        r = cls._throttled_get(params)
        if r is None:
            return []
        try:
            results = r.json().get("search", [])
            return [item.get("id") for item in results if item.get("id")]
        except Exception as e:
            print(f"⚠️ Wikidata (parsing recherche) erreur pour '{location_name}' : {e}")
            return []

    @classmethod
    def _get_country_for_entity(cls, entity_id):
        params = {
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "claims",
            "format": "json",
        }
        r = cls._throttled_get(params)
        if r is None:
            return None
        try:
            entities = r.json().get("entities", {})
            entity = entities.get(entity_id, {})
            claims = entity.get("claims", {})
            country_claims = claims.get("P17")
            if not country_claims:
                return None
            country_entity_id = (
                country_claims[0]
                .get("mainsnak", {})
                .get("datavalue", {})
                .get("value", {})
                .get("id")
            )
            if not country_entity_id:
                return None
            return cls._resolve_entity_label(country_entity_id)
        except Exception as e:
            print(f"⚠️ Wikidata (parsing pays) erreur pour '{entity_id}' : {e}")
            return None

    @classmethod
    def _resolve_entity_label(cls, entity_id):
        params = {
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "labels",
            "languages": "fr|en",
            "format": "json",
        }
        r = cls._throttled_get(params)
        if r is None:
            return None
        try:
            entities = r.json().get("entities", {})
            entity = entities.get(entity_id, {})
            labels = entity.get("labels", {})
            label = labels.get("fr") or labels.get("en")
            if label:
                return label.get("value")
        except Exception as e:
            print(f"⚠️ Wikidata (parsing label) erreur : {e}")
        return None

    @classmethod
    def get_real_country(cls, location_name, hint_country=None):
        if not location_name:
            return None

        cache_key = f"{location_name.strip().lower()}|{(hint_country or '').lower()}"
        if cache_key in cls.CACHE:
            return cls.CACHE[cache_key]

        entity_ids = cls._search_entity_ids(location_name, limit=3)
        if not entity_ids:
            cls.CACHE[cache_key] = None
            return None

        candidates_countries = []
        for entity_id in entity_ids:
            country = cls._get_country_for_entity(entity_id)
            candidates_countries.append(country)

            if hint_country and country and _countries_match(hint_country, country):
                cls.CACHE[cache_key] = country
                return country

        result = next((c for c in candidates_countries if c), None)
        cls.CACHE[cache_key] = result
        return result


def _normalize_country_text(text):
    return re.sub(r"[^a-z]", "", str(text or "").lower())


def _countries_match(declared, real):
    if not declared or not real:
        return True
    d = _normalize_country_text(declared)
    r = _normalize_country_text(real)
    return d == r or d in r or r in d


_RATE_LIMIT_NUMS_RE = re.compile(r"Limit[:\s]+(\d+),?\s*Requested[:\s]+(\d+)", re.IGNORECASE)
SAFETY_MARGIN_TOKENS = 400


def _parse_rate_limit_numbers(err_str):
    match = _RATE_LIMIT_NUMS_RE.search(err_str)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _is_rate_limit_error(err_str):
    err_lower = err_str.lower()
    return (
        "rate_limit_exceeded" in err_lower
        or "tokens per minute" in err_lower
        or "tpm" in err_lower
        or "requests per minute" in err_lower
        or "rpm" in err_lower
    )


def _estimate_tokens(text):
    if not text:
        return 0
    return max(1, len(text) // 3)


def _estimate_prompt_tokens(messages):
    return sum(_estimate_tokens(m.get("content", "")) for m in messages)


class ContentBrain:
    def __init__(self):
        pass

    def _extract_content(self, response):
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            raise ValueError(f"Réponse inattendue de Groq: {response}")

        choice0 = choices[0]
        message = getattr(choice0, "message", None)
        if message is None and isinstance(choice0, dict):
            message = choice0.get("message")

        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        if content is None and isinstance(choice0, dict):
            content = choice0.get("content")

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text") or item.get("content") or "")
                elif isinstance(item, str):
                    parts.append(item)
            content = "".join(parts).strip()

        if not content:
            finish_reason = getattr(choice0, "finish_reason", None)
            if finish_reason is None and isinstance(choice0, dict):
                finish_reason = choice0.get("finish_reason")

            if finish_reason == "length":
                raise ValueError("Contenu vide de Groq : budget de tokens épuisé (finish_reason='length'). Augmenter max_completion_tokens.")

            raise ValueError(f"Contenu vide de Groq: {response}")

        return content.strip()

    def _call_with_fallback(self, messages, temperature=1.0, json_mode=False,
                             max_completion_tokens=3000, hard_token_cap=7500):
        
        # Configuration dynamique des fournisseurs (sans coder les modèles en dur)
        providers = [
            {
                "name": "Groq",
                "model": GROQ_MODEL,
                "base_url": "https://api.groq.com/openai/v1",
                "api_key": os.getenv("GROQ_API_KEY")
            },
            {
                "name": "OpenRouter (Fallback 1)",
                "model": OPENROUTER_FALLBACK_MODEL_1,
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": os.getenv("OPENROUTER_API_KEY")
            },
            {
                "name": "OpenRouter (Fallback 2)",
                "model": OPENROUTER_FALLBACK_MODEL_2,
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": os.getenv("OPENROUTER_API_KEY")
            }
        ]

        last_error = None
        prompt_tokens_est = _estimate_prompt_tokens(messages)
        
        for provider in providers:
            if not provider["api_key"]:
                print(f"⚠️ Clé API manquante pour {provider['name']}, passage au suivant.")
                continue
                
            # Instanciation dynamique du client pour ce fournisseur
            client = OpenAI(base_url=provider["base_url"], api_key=provider["api_key"])
            current_max_tokens = max_completion_tokens
            
            available = hard_token_cap - prompt_tokens_est - SAFETY_MARGIN_TOKENS
            if current_max_tokens > available:
                adjusted = max(500, available)
                if adjusted < current_max_tokens:
                    print(f"ℹ️ Budget de sortie réduit préventivement : {current_max_tokens} -> {adjusted}")
                    current_max_tokens = adjusted

            for attempt in range(3):
                try:
                    kwargs = {
                        "model": provider["model"],
                        "messages": messages,
                        "temperature": temperature,
                        "max_completion_tokens": current_max_tokens,
                    }
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}

                    response = client.chat.completions.create(**kwargs)
                    content = self._extract_content(response)
                    print(f"✅ Réponse obtenue via {provider['name']} ({provider['model']})")
                    time.sleep(1)
                    return content

                except Exception as e:
                    err_str = str(e)
                    print(f"⚠️ Échec avec {provider['name']} (Tentative {attempt + 1}/3): {e}")
                    last_error = e
                    
                    # DÉTECTION DE LA CENSURE (ERREUR 400)
                    if "invalid_request_error" in err_str and ("I'm sorry" in err_str or "I’m sorry" in err_str or "failed_generation" in err_str):
                        print(f"🚫 Blocage de sécurité détecté sur {provider['name']}. Bascule immédiate vers le fallback...")
                        break # Quitte la boucle 'attempt' et passe au fournisseur suivant (OpenRouter)
                        
                    # GESTION DES LIMITES DE TOKENS
                    if isinstance(e, ValueError) and "budget de tokens épuisé" in err_str:
                        current_max_tokens = max(500, min(current_max_tokens + 1500, available))
                        time.sleep(2)
                        continue

                    rate_limit_nums = _parse_rate_limit_numbers(err_str)
                    if rate_limit_nums:
                        limit, requested = rate_limit_nums
                        overage = requested - limit
                        current_max_tokens = max(500, current_max_tokens - overage - SAFETY_MARGIN_TOKENS)
                        time.sleep(1)
                    elif _is_rate_limit_error(err_str):
                        wait_time = 10 if "OpenRouter" in provider["name"] else 60
                        print(f"⏳ Limite de débit atteinte sur {provider['name']}, attente de {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        current_max_tokens = max(500, min(current_max_tokens + 1000, available))
                        time.sleep(4)
                        
            print(f"❌ Échec définitif pour {provider['name']} après 3 tentatives.")

        # Si la boucle se termine sans 'return', tous les providers ont échoué
        raise RuntimeError(f"Brain Error (Script): Échec total après utilisation de tous les modèles (Groq + OpenRouter). Dernière erreur: {last_error}")

    def _call_json_with_retry(self, messages, temperature=1.0, max_json_retries=2,
                              max_completion_tokens=6000, hard_token_cap=7500):
        last_error = None

        for attempt in range(max_json_retries):
            content = self._call_with_fallback(
                messages,
                temperature=temperature,
                json_mode=True,
                max_completion_tokens=max_completion_tokens,
                hard_token_cap=hard_token_cap,
            )
            try:
                data = json.loads(_clean_json_response(content))
                return data
            except json.JSONDecodeError as e:
                last_error = e
                print(f"⚠️ JSON malformé reçu (tentative {attempt + 1}/{max_json_retries}), "
                      f"nouvelle tentative avec budget de tokens augmenté...")
                max_completion_tokens = min(max_completion_tokens + 1000, 7000)

        raise ValueError(f"Impossible d'obtenir un JSON valide après {max_json_retries} tentatives : {last_error}")

    def get_trending_topic(self, previous_stats_list=None):
        stats_instruction = _format_stats_instruction(previous_stats_list, label="sujet")
        used_topics = load_topic_history()

        messages = [
            {"role": "system", "content": (
                "Tu es un créateur de contenu ultra-créatif et totalement imprévisible. "
                "Ta force est la diversité absolue : tu ne recycles jamais tes structures de phrases, "
                "ton vocabulaire ou tes angles d'attaque. "
                "Reponds uniquement avec un seul titre en francais, une seule ligne, sans guillemets, maximum 18 mots. "
                "Ne montre jamais ton raisonnement. "
                f"{ACCENT_INSTRUCTION} {NO_META_AI_INSTRUCTION} {VERACITY_INSTRUCTION}"
            )},
            {"role": "user", "content": (
                "Donne un sujet viral totalement inédit pour TikTok en français, "
                "portant sur un fait historique URBAIN REEL et verifiable, très peu connu du grand public, "
                "SITUÉ STRICTEMENT EN FRANCE (idéalement des secrets parisiens ou des légendes urbaines locales). "
                "INSTRUCTION DE DIVERSITÉ : Explore exclusivement le patrimoine français "
                "(catacombes, monuments oubliés, lois absurdes locales, anecdotes architecturales, affaires classées). "
                "Alterne radicalement tes formats de titres à chaque fois : utilise parfois une question provocante, "
                "parfois une affirmation brute, un paradoxe frappant, ou une simple anecdote ciblée."
                + stats_instruction
                + (
                    "\n\nSUJETS DEJA TRAITES (Casse tes habitudes : propose un univers, un vocabulaire et un format "
                    "radicalement différents de ce qui figure dans cette liste) :\n- " 
                    + "\n- ".join(used_topics[-30:]) if used_topics else ""
                )
            )},
        ]

        last_topic = ""
        for attempt in range(5):
            content = self._call_with_fallback(
                messages, temperature=0.9, max_completion_tokens=2000
            )
            topic = _clean_single_line_title(content)
            last_topic = topic

            if _contains_ai_mention(topic):
                print(f"⚠️ Sujet rejeté (mention IA détectée, tentative {attempt + 1}) : {topic}")
                continue

            if not (topic and 4 <= len(topic.split()) <= 18):
                print(f"⚠️ Sujet invalide genere (tentative {attempt + 1}) : {topic}")
                continue

            is_dup, matched = is_duplicate_topic(topic, used_topics)
            if is_dup:
                print(f"⚠️ Sujet rejeté (doublon de '{matched}', tentative {attempt + 1}) : {topic}")
                continue

            return topic

        raise ValueError(f"Impossible d'obtenir un sujet valide et unique apres 5 tentatives : {last_topic}")

    def refine_topic_angle(self, raw_topic):
        messages = [
            {"role": "system", "content": (
                "Tu reformules le sujet en un titre accrocheur, sans changer le theme "
                "ni la localisation geographique reelle du fait evoque. "
                "Reponds uniquement avec le titre reformule, sans montrer ton raisonnement. "
                f"{ACCENT_INSTRUCTION} {NO_META_AI_INSTRUCTION} {VERACITY_INSTRUCTION}"
            )},
            {"role": "user", "content": f"Sujet brut / trend repere: {raw_topic}"},
        ]
        content = self._call_with_fallback(
            messages, temperature=0.8, max_completion_tokens=2000
        )
        refined = _clean_single_line_title(content)

        if not refined or _contains_ai_mention(refined):
            print(f"⚠️ Reformulation rejetée (vide ou mention IA détectée) : '{refined}' → on garde le sujet brut.")
            return _clean_single_line_title(raw_topic)

        return refined

    def generate_video_search_query(self, topic):
        messages = [
            {"role": "system", "content": (
                "Tu génères une requête de recherche visuelle en anglais, 6 mots max, "
                "sans phrase, sans raisonnement visible -- reponds directement avec la "
                "requete finale. Inclure des termes comme photorealistic, historical "
                "documentary, real photography, dark mysterious atmosphere. INTERDICTION "
                "ABSOLUE d'utiliser les mots CGI, 3D, render ou Unreal Engine."
            )},
            {"role": "user", "content": f"Sujet : {topic}"},
        ]
        content = self._call_with_fallback(
            messages, temperature=0.7, max_completion_tokens=1500
        )
        query = _clean_single_line_title(content).replace('"', '')

        if not query:
            return "photorealistic historical documentary dark atmosphere"
        return query

    def generate_hook_variants(self, topic, n=5, previous_stats_list=None):
        stats_instruction = _format_stats_instruction(previous_stats_list, label="hooks")
        prompt = f"""
SUJET:
{topic}

GENERE {n} hooks viraux en francais.

RETURNS JSON:
{{
  "hooks": [
    {{
      "text": "hook",
      "pattern": "question",
      "raison": "..."
    }}
  ]
}}

{stats_instruction}
"""
        messages = [
            {"role": "system", "content": (
                f"Tu produis uniquement du JSON valide avec exactement {n} hooks, "
                "sans aucun texte ni raisonnement en dehors du JSON. "
                f"{ACCENT_INSTRUCTION} {NO_META_AI_INSTRUCTION}"
            )},
            {"role": "user", "content": prompt},
        ]

        data = self._call_json_with_retry(
            messages, temperature=1.1, max_completion_tokens=4000
        )

        hooks = data.get("hooks")
        if not isinstance(hooks, list):
            raise ValueError("Champ hooks invalide.")

        normalized_hooks = []
        for h in hooks:
            if isinstance(h, str):
                text = h.strip()
                if text and not _contains_ai_mention(text):
                    normalized_hooks.append({"text": text, "pattern": "question", "raison": ""})
            elif isinstance(h, dict):
                text = str(h.get("text", "")).strip()
                if text and not _contains_ai_mention(text):
                    normalized_hooks.append({
                        "text": text,
                        "pattern": str(h.get("pattern", "question")).strip(),
                        "raison": str(h.get("raison", "")).strip()
                    })

        if len(normalized_hooks) < n:
            raise ValueError(f"Nombre de hooks invalide: {len(normalized_hooks)} au lieu de {n}.")

        normalized_hooks = normalized_hooks[:n]
        if not normalized_hooks:
            normalized_hooks = [{"text": topic, "pattern": "default", "raison": ""}]
        return normalized_hooks

    def propose_real_case(self, topic):
        if not GROUNDING_AVAILABLE:
            return {"case_name": None, "wiki_query": None, "source": None}

        hint_country = _guess_country_hint(topic) or "France"

        messages = [
            {"role": "system", "content": (
                "Tu proposes un cas historique REEL et verifiable, peu connu du "
                "grand public, correspondant exactement au sujet donne. "
                "ATTENTION : sois tres precis sur le lieu exact -- il existe "
                "souvent plusieurs legendes/lieux similaires dans differentes "
                "villes/regions (ex: plusieurs 'ponts du diable' en France a "
                "des endroits differents). Verifie mentalement que le lieu "
                "que tu proposes correspond exactement aux details du sujet "
                "(region, nom propre, contexte). "
                "Reponds UNIQUEMENT avec le nom propre exact du lieu/evenement/ "
                "personnage principal (celui qui a un article Wikipedia), sans "
                "phrase, sans guillemets, sans raisonnement visible, une seule ligne."
            )},
            {"role": "user", "content": f"Sujet : {topic}\n\nDonne le nom exact du cas reel principal a developper."},
        ]

        content = self._call_with_fallback(
            messages, temperature=0.3, max_completion_tokens=1500
        )
        case_name = _clean_single_line_title(content)

        if not case_name:
            return {"case_name": None, "wiki_query": None, "source": None}

        if hint_country:
            real_country = WikidataChecker.get_real_country(case_name, hint_country=hint_country)
            if real_country and not _countries_match(hint_country, real_country):
                print(f"⚠️ Lieu propose '{case_name}' semble incorrect "
                      f"(Wikidata: {real_country}, attendu: {hint_country}). "
                      f"Nouvelle tentative avec consigne renforcee...")

                messages_retry = [
                    {"role": "system", "content": (
                        "Tu proposes un cas historique REEL, peu connu, "
                        f"situe EXACTEMENT en {hint_country} (pas ailleurs). "
                        "Reponds UNIQUEMENT avec le nom propre exact du lieu, "
                        "sans phrase, sans guillemets, sans raisonnement visible, une seule ligne."
                    )},
                    {"role": "user", "content": f"Sujet : {topic}"},
                ]
                content_retry = self._call_with_fallback(
                    messages_retry, temperature=0.3, max_completion_tokens=1500
                )
                case_name_retry = _clean_single_line_title(content_retry)
                if case_name_retry:
                    case_name = case_name_retry

        source = fetch_grounding_source(case_name, hint_country=hint_country)

        return {
            "case_name": case_name,
            "wiki_query": case_name,
            "source": source,
        }

    def generate_script(self, topic, chosen_hook=None):
        return self.generate_script_with_target(topic, scene_count=11, chosen_hook=chosen_hook)

    def generate_script_with_target(self, topic, scene_count=11, chosen_hook=None, max_fact_check_retries=2):
        candidate_counts = []
        sc = scene_count
        while sc >= 6 and len(candidate_counts) < 3:
            candidate_counts.append(sc)
            sc -= 3
        if not candidate_counts:
            candidate_counts = [scene_count]

        last_error = None
        for idx, candidate_count in enumerate(candidate_counts):
            if idx > 0:
                print(f"⚠️ Nouvelle tentative avec un nombre de scenes reduit : "
                      f"{candidate_count} (au lieu de {scene_count}) pour "
                      f"respecter le budget de tokens Groq.")
            try:
                return self._generate_script_attempt(
                    topic, candidate_count, chosen_hook, max_fact_check_retries
                )
            except (RuntimeError, ValueError) as e:
                last_error = e
                print(f"❌ Echec de generation avec {candidate_count} scenes : {e}")
                continue

        raise RuntimeError(
            f"Impossible de generer un script meme en reduisant le nombre "
            f"de scenes ({candidate_counts}). Derniere erreur : {last_error}"
        )

    def _generate_script_attempt(self, topic, scene_count, chosen_hook, max_fact_check_retries):
        hook_instruction = f'La scene 1 doit reprendre ce hook : "{chosen_hook}"' if chosen_hook else "Scene 1: Accroche choc."

        hint_country = _guess_country_hint(topic)

        grounding = self.propose_real_case(topic)
        source = grounding.get("source")

        if source:
            extract_text = source['extract'][:1200] if 'extract' in source else ""
            grounding_block = f"""

SOURCE VERIFIEE OBLIGATOIRE (Wikipedia, {source['lang']}) :
Titre reel : {source['title']}
Extrait de reference : \"\"\"{extract_text}\"\"\"

REGLE ABSOLUE : tu dois baser TOUS les faits du script (dates, lieux, noms,
deroule des evenements) UNIQUEMENT sur cet extrait. Interdiction d'ajouter
un fait, un detail chiffre ou un nom qui n'apparait pas dans cet extrait.
Tu peux reformuler et dramatiser le style, mais pas inventer de contenu
factuel supplementaire.

CORRECTIF (IMAGES CONCRETES) : Pour la clé 'event_context', si une scene
decrit un evenement precis et date mentionne dans l'extrait ci-dessus
(ex: un incendie, une destruction, une decouverte), resume ce contexte en
quelques mots factuels visuellement exploitables pour generer une image
(ex: "nocturnal fire, monastery ruins in flames, november 2025"). Si la scene ne decrit pas d'evenement precis et date, laisse ce champ vide ("").
"""
            print(f"🔗 Script ancre sur la source Wikipedia : '{source['title']}'")
        else:
            # CORRECTIF: On force le modèle à parler de légende pour apaiser le Fact-Checker
            grounding_block = """
ATTENTION - MODE LÉGENDE URBAINE :
Aucune source historique stricte n'a été trouvée pour ce sujet.
Pour éviter de générer des faits historiquement faux, tu DOIS obligatoirement 
présenter cette histoire comme une "légende locale", une "rumeur persistante" 
ou un "mythe urbain". N'affirme pas que c'est une vérité historique absolue.
"""
            print("⚠️ Aucune source Wikipedia trouvee. Le script sera oriente comme une légende urbaine pour passer le fact-checking.")

        base_prompt = f"""
SUJET:
{topic}

{hook_instruction}
{grounding_block}

REGLES STRICTES DE NARRATION ET VISUEL (POUR ÉVITER LES INTROS VIDES ET LA 3D) :
1. Interdiction de faire de longs discours d'introduction ou des bandes-annonces vides ("Nous allons vous raconter...").
2. Dès la scène 2, entre DIRECTEMENT dans le vif du sujet en racontant de vrais faits historiques, des détails précis ou une anecdote concrète et surprenante.
3. Le milieu de la vidéo doit développer l'histoire en profondeur (les faits, les mystères, les rebondissements).
4. Les dernières scènes doivent apporter une conclusion claire ou une révélation, pas s'arrêter en plein milieu.
5. CORRECTION VISUELLE MAJEURE pour 'image_prompt' : TU DOIS OBLIGATOIREMENT L'ÉCRIRE EN ANGLAIS. Génère une description STRICTEMENT VISUELLE ET MATÉRIELLE de 15 mots maximum. Décris uniquement ce que l'on voit physiquement (objets concrets, architecture, textures, lumière). Interdiction absolue d'utiliser des concepts abstraits (legend, myth, secret, story, escape, history) ou des termes de synthèse (CGI, 3D). EXEMPLE PARFAIT : "narrow damp stone corridor, flickering wooden torch, muddy ground, cinematic lighting".
6. Si la scène se déroule dans un vrai lieu (monument, ville, château, île, etc.), donne le nom précis et complet dans 'location_name' (ex: "Château de Chambord", "Église Saint-Pierre d'Oron" et non juste "Église Saint-Pierre") ET le pays reel dans 'location_country' (ex: "France"). Si c'est juste de l'ambiance ou abstrait, laisse les deux vides ("").
7. Pour la clé 'voice_type', choisis "narrator" pour l'ambiance globale/les faits, ou "witness" pour dynamiser (citations, avis, phrases choc). Alterne intelligemment pour garder l'audience captivée.
8. Interdiction absolue de mentionner l'intelligence artificielle, l'IA, un algorithme, ou tout aspect meta lié a la creation de la video. Chaque scene doit parler uniquement du mystere/de l'histoire reelle, jamais de la maniere dont la video a ete produite.
9. Respecte scrupuleusement les exigences de veracite historique donnees dans les instructions systeme (lieux/faits reels, pays exact, pas d'invention).
10. Pour la clé 'scene_type', choisis "specific" if la scène décrit un événement, un lieu ou un objet historique précis (ex: une épave, une momie, un manuscrit). Choisis "generic" si la scène décrit une ambiance, un paysage naturel ou une émotion (ex: vagues sombres, forêt brumeuse).
11. LECTURE AUDIO : Le texte sera lu par une synthèse vocale. N'utilise JAMAIS de chiffres romains. Écris-les obligatoirement EN TOUTES LETTRES (ex: écris "vingtième siècle" au lieu de "XXe siècle", "Louis quatorze" au lieu de "Louis XIV").
12. IMPORTANT POUR 'stock_search' (Recherche de vidéos) : Ne demande JAMAIS de lieux géographiques précis, de noms propres ou de graphiques. Fournis TOUJOURS un mot-clé très générique, descriptif, d'ambiance et OBLIGATOIREMENT EN ANGLAIS. (Exemple : au lieu de 'Mairie de Sarlat', écris 'old medieval village building').
13. RYTHME ULTRA-COURT : Pour garantir le dynamisme de la vidéo, le 'text' de chaque scène doit être très court (UNE SEULE PHRASE de 10 à 15 mots maximum). La vidéo changera ainsi d'image toutes les 3 secondes.
14. Pour la clé 'event_context' (optionnelle) : voir instruction detaillee ci-dessus si une source verifiee est fournie. Sinon, laisse ce champ vide ("") sauf si le sujet lui-meme mentionne clairement un eventement precis et date (incendie, destruction, decouverte) a illustrer concretement.
15. Ne montre jamais ton raisonnement interne : reponds directement avec le JSON final, sans aucun texte avant ou apres.
16. RE-HOOK OBLIGATOIRE : la scene situee approximativement au tiers du
script (ex: scene 4 sur 11) doit contenir une phrase de rupture qui relance
la curiosite (ex: 'Mais ce n'est pas la le plus troublant...', 'Voici ou
l'histoire prend un tournant...') pour retenir les spectateurs qui commencent
a decrocher.
17. CTA FINAL : genere un champ 'closing_cta' independant dans le JSON
(au meme niveau que 'title', 'visual_identity', 'audio_profile'), une
phrase courte (8-10 mots) qui invite a s'abonner en creant une attente
specifique pour la prochaine video (ex: 'Abonne-toi, demain je devoile un
secret encore plus trouble'), jamais un CTA generique type 'like et
abonne-toi'.
18. Pour la clé 'mood', choisis OBLIGATOIREMENT parmi cette liste : "ominous", "intriguing", "tense", "awe", "scientific", "investigation".
19. Pour la clé 'role', choisis OBLIGATOIREMENT parmi cette liste : "hook", "tension", "context", "value", "escalation", "reveal", "cta".

CONTRAINTE CRITIQUE ET NON NEGOCIABLE SUR LE FORMAT :
Tu DOIS retourner EXACTEMENT {scene_count} scenes dans le tableau 'scenes' -- ni plus, ni moins.
Compte precisement le nombre d'elements avant de repondre. Si tu ne peux pas
developper {scene_count} scenes avec suffisamment de matiere, repartis le
contenu disponible sur EXACTEMENT {scene_count} scenes plus courtes plutot
qu'en generer moins. Ne tronque JAMAIS ta reponse JSON avant d'avoir
ecrit les {scene_count} scenes completes et la fermeture correcte du JSON.
Reste CONCIS sur chaque champ texte pour ne pas depasser le budget de
tokens disponible tout en couvrant les {scene_count} scenes en entier.

Retourne un JSON avec les clés :
title, visual_identity, audio_profile, scenes.
Chaque scene dans le tableau 'scenes' doit contenir :
id, text, voice_direction, pause_after_ms, stock_search, image_prompt, location_name, location_country, voice_type, mood, role, scene_type, event_context.
"""
        estimated_tokens_needed = min(scene_count * 280 + 1200, 6000)
        correction_feedback = ""

        for fact_check_attempt in range(max_fact_check_retries + 1):
            prompt = base_prompt + correction_feedback

            messages = [
                {"role": "system", "content": (
                    f"Tu produis uniquement du JSON valide. La cle scenes contient EXACTEMENT {scene_count} scenes, "
                    f"ni plus ni moins -- c'est une contrainte absolue et non negociable. "
                    f"Ne montre jamais ton raisonnement, reponds directement avec le JSON final. "
                    f"{ACCENT_INSTRUCTION} {NO_META_AI_INSTRUCTION} {VERACITY_INSTRUCTION}"
                )},
                {"role": "user", "content": prompt},
            ]

            data = self._call_json_with_retry(
                messages,
                temperature=0.7,
                max_completion_tokens=estimated_tokens_needed,
            )

            scenes = data.get("scenes", [])
            if not isinstance(scenes, list):
                scenes = []

            scene_count_issue = None
            if len(scenes) != scene_count:
                scene_count_issue = (
                    f"Le JSON genere contient {len(scenes)} scenes au lieu des "
                    f"{scene_count} scenes exactement demandees. Il faut "
                    f"generer EXACTEMENT {scene_count} scenes, ni plus ni moins."
                )

            if scene_count_issue:
                print(f"⚠️ Comptage de scenes incorrect (tentative {fact_check_attempt + 1}) : {scene_count_issue}")

                if fact_check_attempt < max_fact_check_retries:
                    correction_feedback = f"""

CORRECTION OBLIGATOIRE :
{scene_count_issue}
Regenere le script complet avec EXACTEMENT {scene_count} scenes cette fois-ci.
Reste concis sur chaque champ texte pour respecter le budget de tokens. NE T'ARRETE PAS AVANT LA FIN.
"""
                    continue
                else:
                    raise ValueError(
                        f"Impossible d'obtenir {scene_count} scenes apres "
                        f"{max_fact_check_retries + 1} tentatives (dernier essai : {len(scenes)} scenes)."
                    )

            for idx, scene in enumerate(scenes, start=1):
                if isinstance(scene, dict):
                    scene.setdefault("id", idx)
                    scene.setdefault("voice_direction", "French premium narrator, calm, elegant, intriguing, controlled pacing")
                    scene.setdefault("pause_after_ms", 300)
                    scene.setdefault("stock_search", "cinematic vertical background")
                    scene.setdefault("image_prompt", "Vertical 9:16 cinematic scene")
                    scene.setdefault("location_name", "")
                    scene.setdefault("location_country", "")
                    scene.setdefault("voice_type", "narrator")
                    scene.setdefault("mood", "intriguing")
                    scene.setdefault("role", "value")
                    scene.setdefault("scene_type", "generic")
                    scene.setdefault("event_context", "")

            try:
                self._validate_script(data, scene_count, topic)
            except ValueError as e:
                print(f"⚠️ Erreur de validation (tentative {fact_check_attempt + 1}) : {e}")
                if fact_check_attempt < max_fact_check_retries:
                    correction_feedback = f"""

CORRECTION OBLIGATOIRE :
Le script precedent a echoue a la validation : {e}
Corrige ce probleme specifique et regenere un script complet et valide avec
EXACTEMENT {scene_count} scenes.
"""
                    continue
                else:
                    raise

            geo_issue = self._check_geography_consistency(topic, data["scenes"])
            wikidata_issues = self._check_wikidata_locations(data["scenes"], hint_country=hint_country)
            fact_check_result = self._fact_check_script(topic, data, grounding_source=source)

            issues = []
            if geo_issue:
                issues.append(geo_issue)
            issues.extend(wikidata_issues)
            issues.extend(fact_check_result.get("issues", []))

            all_consistent = (
                not geo_issue
                and not wikidata_issues
                and fact_check_result.get("is_consistent", True)
            )

            if all_consistent:
                if fact_check_attempt > 0:
                    print(f"✅ Script valide apres correction (tentative {fact_check_attempt + 1}).")
                return data

            if fact_check_attempt < max_fact_check_retries:
                print(f"⚠️ Incoherences factuelles/geographiques detectees (tentative {fact_check_attempt + 1}) :")
                for issue in issues:
                    print(f"    - {issue}")

                correction_feedback = f"""

CORRECTION OBLIGATOIRE :
Le script precedent contenait les incoherences suivantes, a corriger imperativement :
{chr(10).join(f"- {i}" for i in issues)}
Regenere un script totalement coherent avec des faits REELS et bien localises,
en respectant strictement les memes regles{" et la source verifiee fournie" if source else ""}.
IMPORTANT : garde EXACTEMENT {scene_count} scenes.
"""
            else:
                print(f"❌ Incoherences factuelles persistantes apres {max_fact_check_retries + 1} tentatives, "
                      f"video generee malgre tout (verification manuelle recommandee) :")
                for issue in issues:
                    print(f"    - {issue}")
                return data

        return data

    def _check_geography_consistency(self, topic, scenes):
        if not _topic_claims_french_location(topic):
            return None

        for scene in scenes:
            country = str(scene.get("location_country", "")).strip().lower()
            location = str(scene.get("location_name", "")).strip()
            if country and "france" not in country and "français" not in country:
                return (
                    f"Le sujet annonce une localisation francaise, mais la scene "
                    f"mentionnant '{location}' indique le pays '{country}', ce qui "
                    f"est incoherent avec le sujet annonce."
                )
        return None

    def _check_wikidata_locations(self, scenes, hint_country=None):
        issues = []
        not_found_locations = []

        for scene in scenes:
            location_name = str(scene.get("location_name", "")).strip()
            declared_country = str(scene.get("location_country", "")).strip()

            if not location_name or not declared_country:
                continue

            real_country = WikidataChecker.get_real_country(location_name, hint_country=hint_country)

            if real_country is None:
                not_found_locations.append(location_name)
                continue

            if not _countries_match(declared_country, real_country):
                issues.append(
                    f"Wikidata indique que '{location_name}' se trouve reellement "
                    f"en '{real_country}', mais le script declare '{declared_country}'. "
                    f"Utilise un nom de lieu plus specifique/complet ou corrige la "
                    f"localisation."
                )
                print(f"❌ Wikidata mismatch : '{location_name}' est en '{real_country}' "
                      f"(declare : '{declared_country}')")
            else:
                print(f"✅ Wikidata confirme : '{location_name}' est bien en '{real_country}'")

        if not_found_locations:
            print(f"ℹ️ Wikidata : {len(not_found_locations)} lieu(x) introuvable(s), "
                  f"verification ignoree ({', '.join(not_found_locations)}).")

        return issues

    def _fact_check_script(self, topic, script_data, grounding_source=None):
        scenes_summary = "\n".join(
            f"- Scene {s.get('id')} [{s.get('location_name', 'aucun lieu')} / "
            f"{s.get('location_country', 'pays non precise')}] : {s.get('text', '')[:200]}"
            for s in script_data.get("scenes", [])
        )

        if grounding_source:
            extract_text = grounding_source['extract'][:1200]
            prompt = f"""
Tu es un fact-checker rigoureux. Compare le SCRIPT ci-dessous a la SOURCE DE
REFERENCE (extrait Wikipedia reel) et detecte UNIQUEMENT les affirmations du
script qui CONTREDISENT ou AJOUTENT un fait absent de la source (date, nom,
lieu, evenement invente).

SOURCE DE REFERENCE ({grounding_source['title']}) :
\"\"\"{extract_text}\"\"\"

RESUME DU SCRIPT GENERE :
{scenes_summary}

Retourne UNIQUEMENT du JSON valide, sans raisonnement visible :
{{
  "is_consistent": true/false,
  "issues": ["fait du script absent ou contradictoire avec la source : ..."]
}}

Ne signale PAS de simples reformulations/dramatisations fideles a la source.
"""
        else:
            prompt = f"""
Tu es un fact-checker STRICT. Tu ne signales QUE des erreurs factuelles
objectives et verifiables, jamais des critiques de style ou de manque de
details.

SUJET DE LA VIDEO :
{topic}

RESUME DES SCENES :
{scenes_summary}

Signale UNIQUEMENT :
- Un nom de lieu, personnage ou organisation manifestement invente (n'existe
  pas du tout, pas juste "peu documente")
- Une date historique explicitement fausse et verifiable comme telle
- Une contradiction factuelle directe entre deux scenes (ex: un evenement
  date differemment a deux endroits)

NE SIGNALE JAMAIS :
- Le manque de details ou de precision ("les informations sont trop
  generales", "pas assez de details sur X")
- Le style narratif, le rythme, ou la construction du recit
- L'absence de preuves pour des legendes/rumeurs (une legende reste une
  legende, ce n'est pas une erreur factuelle si elle est presentee comme
  telle)
- Les problemes de geographie/pays (verifies separement par une autre methode)

Retourne UNIQUEMENT du JSON valide, sans raisonnement visible :
{{
  "is_consistent": true/false,
  "issues": ["description precise et actionnable du probleme 1", "..."]
}}

Si tu as le moindre doute, ne signale RIEN (is_consistent: true, issues: []).
"""

        messages = [
            {"role": "system", "content": (
                "Tu produis uniquement du JSON valide, factuel, rigoureux et "
                "minimaliste (peu de faux positifs), sans raisonnement visible."
            )},
            {"role": "user", "content": prompt},
        ]

        try:
            data = self._call_json_with_retry(
                messages, temperature=0.1, max_json_retries=1,
                max_completion_tokens=2500
            )
            if not isinstance(data, dict):
                return {"is_consistent": True, "issues": []}
            return {
                "is_consistent": bool(data.get("is_consistent", True)),
                "issues": data.get("issues", []) if isinstance(data.get("issues"), list) else [],
            }
        except Exception as e:
            print(f"⚠️ Fact-check LLM impossible (erreur technique), on continue sans blocage : {e}")
            return {"is_consistent": True, "issues": []}

    def _validate_script(self, data, scene_count, topic):
        scenes = data.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError("La reponse ne contient pas de tableau scenes.")
        if len(scenes) != scene_count:
            raise ValueError(f"Nombre de scenes invalide : {len(scenes)} au lieu de {scene_count}.")

        allowed_roles = {"hook", "tension", "context", "value", "escalation", "reveal", "cta"}

        # Normalise les variantes courantes renvoyées par le LLM (casse, espaces,
        # accents/pluriels) vers les valeurs canoniques attendues, pour éviter
        # que toute déviation mineure ne retombe systématiquement sur "intriguing".
        mood_aliases = {
            "ominous": "ominous", "sombre": "ominous", "menaçant": "ominous", "menacant": "ominous",
            "intriguing": "intriguing", "mystérieux": "intriguing", "mysterieux": "intriguing",
            "tense": "tense", "tendu": "tense", "tension": "tense",
            "awe": "awe", "impressionnant": "awe", "fascinant": "awe",
            "scientific": "scientific", "scientifique": "scientific",
            "investigation": "investigation", "enquête": "investigation", "enquete": "investigation",
        }

        for scene in scenes:
            if not isinstance(scene, dict):
                raise ValueError("Une scène n'est pas un dictionnaire valide.")
            if not scene.get("text"):
                raise ValueError(f"Scene {scene.get('id')} : text manquant.")

            if _contains_ai_mention(scene.get("text", "")):
                raise ValueError(
                    f"Scene {scene.get('id')} : mention d'IA/technologie detectee dans le texte "
                    f"('{scene['text'][:80]}...'), regeneration necessaire."
                )

            if not scene.get("voice_direction"):
                scene["voice_direction"] = "French premium narrator, calm, elegant, intriguing, controlled pacing"
            
            pause_after_ms = scene.get("pause_after_ms")
            if not isinstance(pause_after_ms, int):
                scene["pause_after_ms"] = 300
                
            if scene.get("role") not in allowed_roles:
                scene["role"] = "value"

            # --- Normalisation du mood ---
            raw_mood = str(scene.get("mood", "")).strip().lower()
            normalized_mood = mood_aliases.get(raw_mood)
            if normalized_mood:
                if normalized_mood != raw_mood:
                    print(f"      ℹ️ Scene {scene.get('id')} : mood '{raw_mood}' normalisé en '{normalized_mood}'")
                scene["mood"] = normalized_mood
            else:
                print(f"      ⚠️ Scene {scene.get('id')} : mood invalide reçu ('{scene.get('mood')}'), fallback -> 'intriguing'")
                scene["mood"] = "intriguing"

            if not scene.get("stock_search"):
                scene["stock_search"] = "cinematic vertical background"
            if not scene.get("image_prompt"):
                scene["image_prompt"] = "Vertical 9:16 cinematic scene"
            if "location_name" not in scene:
                scene["location_name"] = ""
            if "location_country" not in scene:
                scene["location_country"] = ""
            if scene.get("voice_type") not in {"narrator", "witness"}:
                scene["voice_type"] = "narrator"
            if "scene_type" not in scene or scene.get("scene_type") not in {"generic", "specific"}:
                scene["scene_type"] = "generic"
            if "event_context" not in scene or not isinstance(scene.get("event_context"), str):
                scene["event_context"] = ""

        if not str(data.get("title", "")).strip():
            data["title"] = topic
        if not str(data.get("visual_identity", "")).strip():
            data["visual_identity"] = "Consistent cinematic vertical documentary world."
        if not str(data.get("audio_profile", "")).strip():
            data["audio_profile"] = "French premium narrator, calm, elegant, slightly deep, natural, controlled pacing"