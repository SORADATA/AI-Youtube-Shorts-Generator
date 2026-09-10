import os
import time
import hashlib
import requests


class AIImageGenerator:
    BASE_STYLE = (
        "cinematic realistic documentary still, photorealistic, atmospheric depth, "
        "subtle film grain, dramatic but natural lighting, rich natural textures, "
        "high visual clarity, strictly centered subject, extreme wide shot, vertical 9:16 composition, "
        "subject kept inside the center safe zone, clean space near the top and bottom for captions"
    )

    NEGATIVE_PROMPT = (
        "text, logo, watermark, subtitles, borders, split screen, multiple frames, "
        "deformed hands, extra fingers, duplicate people, "
        "cropped head, blurry face, cartoon, anime"
    )

    # Modèle principal, puis modèle de secours si le premier échoue (500 / timeout)
    PRIMARY_MODEL = "flux"
    FALLBACK_MODEL = "turbo"

    def __init__(self):
        contact = os.getenv("WIKIMEDIA_CONTACT", "https://github.com/tonuser")
        self.headers = {"User-Agent": f"AuraContentPipeline/2.0 ({contact})"}

        print("🤖 Initialisation du générateur d'images IA (Pollinations)")

    def _stable_seed(self, prompt_text, visual_identity=None, variant="base"):
        raw = f"{prompt_text}|{visual_identity or ''}|{variant}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 999999 + 1

    def _build_prompt(self, prompt_text, visual_identity=None, variant=None):
        fixed_parts = [
            prompt_text.strip().rstrip(",."),
            self.BASE_STYLE
        ]

        if visual_identity:
            fixed_parts.append(f"visual continuity: {visual_identity.strip().rstrip(',.')}")

        if variant == "a":
            fixed_parts.append("wider establishing composition, environment clearly visible, stable cinematic framing")
        elif variant == "b":
            fixed_parts.append("closer cinematic framing, more subject detail, same scene continuity, same visual world")

        fixed_parts.append(f"avoid: {self.NEGATIVE_PROMPT}")
        return ", ".join(fixed_parts)

    def _file_is_valid(self, output_path, min_bytes=5000):
        return os.path.exists(output_path) and os.path.getsize(output_path) >= min_bytes

    def _build_url(self, enhanced_prompt, seed, model):
        encoded_prompt = requests.utils.quote(enhanced_prompt, safe="")
        return (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width=1080&height=1920"
            f"&seed={seed}"
            f"&model={model}"
            f"&nologo=true&enhance=true"
        )

    def _try_pollinations(self, prompt_text, output_path, visual_identity=None, seed=None, variant=None,
                           model=None, timeout=45):
        enhanced_prompt = self._build_prompt(prompt_text, visual_identity=visual_identity, variant=variant)
        if seed is None:
            seed = self._stable_seed(prompt_text, visual_identity=visual_identity, variant=variant or "base")
        model = model or self.PRIMARY_MODEL

        api_url = self._build_url(enhanced_prompt, seed, model)

        try:
            response = requests.get(api_url, timeout=timeout, headers=self.headers)
            content_type = response.headers.get("Content-Type", "")

            if response.status_code == 200 and content_type.startswith("image/") and len(response.content) > 5000:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, "wb") as file:
                    file.write(response.content)
                return True

            if response.status_code == 429:
                print("    ⚠️ Limite de requêtes atteinte sur Pollinations.")
            else:
                print(f"    ❌ Erreur Pollinations ({model}) : {response.status_code}, type={content_type}")
            return False

        except requests.RequestException as error:
            print(f"    ❌ Erreur réseau Pollinations ({model}) : {error}")
            return False

    def generate_image(self, prompt_text, output_path, visual_identity=None, retries=2, seed=None, variant=None):
        print(f"🎨 Génération d'une image IA pour : '{prompt_text}'")

        if self._file_is_valid(output_path):
            print(f"    ♻️ Image déjà présente : {output_path}")
            return True

        base_seed = seed or self._stable_seed(prompt_text, visual_identity=visual_identity, variant=variant or "base")

        # Tentative 1 : modèle principal
        # Tentative 2 : modèle de secours (souvent plus stable en cas de 500/erreur serveur)
        # Tentative 3 : modèle principal, seed différent, timeout allongé (au cas où c'était un vrai timeout)
        attempts = [
            {"model": self.PRIMARY_MODEL, "seed": base_seed, "timeout": 45},
            {"model": self.FALLBACK_MODEL, "seed": base_seed, "timeout": 45},
            {"model": self.PRIMARY_MODEL, "seed": base_seed + 1, "timeout": 60},
        ][: retries + 1]

        for attempt_index, attempt_config in enumerate(attempts):
            if attempt_index > 0:
                backoff = 2 * attempt_index
                print(f"    🔄 Tentative {attempt_index + 1}/{len(attempts)} "
                      f"(modèle={attempt_config['model']}, pause {backoff}s)...")
                time.sleep(backoff)

            success = self._try_pollinations(
                prompt_text=prompt_text,
                output_path=output_path,
                visual_identity=visual_identity,
                seed=attempt_config["seed"],
                variant=variant,
                model=attempt_config["model"],
                timeout=attempt_config["timeout"],
            )

            if success and self._file_is_valid(output_path):
                print(f"    ✅ Image sauvegardée (Pollinations/{attempt_config['model']}, "
                      f"seed={attempt_config['seed']}) : {output_path}")
                return True

        print(f"    ❌ Échec définitif pour la génération de l'image : {output_path}")
        return False