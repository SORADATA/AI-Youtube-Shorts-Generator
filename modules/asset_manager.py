from modules.utils.download.utils_assets import load_history
from modules.utils.download.video_provider import VideoProvider
from modules.utils.download.archive_provider import ArchiveProvider
from modules.ai_image import AIImageGenerator


class AssetManager:
    def __init__(self):
        self.history = load_history()
        self.videos = VideoProvider(self.history)
        self.archives = ArchiveProvider(self.history)
        self.ai = AIImageGenerator()

    def _build_structured_prompt(self, query, event_context=None, image_prompt=None):
        """
        Construction d'un prompt structuré 100% en ANGLAIS.
        Les prompts négatifs sont retirés d'ici car ils sont déjà 
        gérés par la classe AIImageGenerator.
        """
        if image_prompt and image_prompt.strip():
            subject = image_prompt.strip()
            # CORRECTION : On passe en anglais ("real location" au lieu de "lieu réel")
            if query and query.strip() and query.strip().lower() not in subject.lower():
                subject = f"{subject}, real location: {query.strip()}"
        else:
            subject = (query or "").strip()

        if event_context:
            subject = f"{subject}, {event_context.strip()}"

        # CORRECTION : Optimisation des modificateurs pour le format vertical (sujet centré)
        composition = "wide-angle documentary shot, strictly centered subject, eye-level perspective"
        lighting = "dramatic natural lighting, moody shadows, cinematic ambient light"
        texture = "visible material details, realistic weathered surfaces, natural imperfections"
        palette = "muted desaturated tones, dark cinematic color grading"
        mood = "tense, mysterious, documentary atmosphere, photojournalistic feel"

        structured_prompt = (
            f"{subject}. {composition}. {lighting}. {texture}. "
            f"{palette}. {mood}. photorealistic, ultra-realistic, "
            f"real-world photography, 8k detail."
        )

        return structured_prompt

    def get_best_asset(self, query, output_path, scene_type="generic", event_context=None, image_prompt=None):
        # ---------------------------------------------------------
        # SCÈNES SPÉCIFIQUES (Lieux réels, personnages, objets)
        # ---------------------------------------------------------
        if scene_type == "specific":
            print(f"🔍 Recherche de la vraie photo historique : '{query}'...")
            if self.archives.get_wikimedia(query, output_path):
                print("🏛️ Vraie archive trouvée !")
                return True, "wiki"

            print(f"🌍 Nouvelle tentative Openverse directe : '{query}'...")
            if self.archives.get_openverse(query, output_path):
                print("🏛️ Archive Openverse trouvée (tentative directe) !")
                return True, "openverse"

            ai_prompt = self._build_structured_prompt(
                query, event_context=event_context, image_prompt=image_prompt
            )

            if event_context:
                print(f"🧠 Archive introuvable. Tentative IA-First contextualisée (prompt structuré).")
            else:
                print(f"🧠 Archive introuvable. Tentative IA-First pour : '{query}' (prompt structuré).")

            if self.ai.generate_image(ai_prompt, output_path):
                return True, "ai"

            print(f"🎬 IA échouée pour '{query}', tentative de secours vidéo générique...")
            generic_fallback_query = "mysterious historical documentary atmosphere"
            if self.videos.fetch_background(generic_fallback_query, output_path):
                return True, "video"

            print(f"❌ Échec total de la récupération d'asset pour : '{query}'")
            return False, "none"

        # ---------------------------------------------------------
        # SCÈNES GÉNÉRIQUES (Ambiance, paysages, émotions)
        # ---------------------------------------------------------
        else:
            print(f"🔍 Recherche vidéo d'ambiance : '{query}'...")
            if self.videos.fetch_background(query, output_path):
                return True, "video"

            fallback_prompt = self._build_structured_prompt(
                query, event_context=event_context, image_prompt=image_prompt
            )
            print(f"🎨 Génération IA de secours (prompt structuré)...")
            if self.ai.generate_image(fallback_prompt, output_path):
                return True, "ai"

            print(f"❌ Échec total de la récupération d'asset pour : '{query}'")
            return False, "none"