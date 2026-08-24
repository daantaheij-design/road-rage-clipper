from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Private access
    app_password: str = "change-me"
    secret_key: str = "insecure-dev-secret-change-me"
    mcp_api_key: str = "insecure-dev-mcp-key-change-me"

    # Anthropic
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5"

    # ElevenLabs
    elevenlabs_api_key: str = ""
    elevenlabs_stt_model: str = "scribe_v2"
    elevenlabs_tts_model: str = "eleven_multilingual_v2"
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    # Voice delivery, all tunable without a code change. Defaults aim for an
    # energetic, expressive modern TikTok/Shorts narrator rather than a flat
    # documentary/news-reader read: lower stability = more emotional
    # range/less monotone, higher style = more exaggerated delivery, speed
    # slightly above 1.0 = a touch of urgency without sounding rushed. Valid
    # ranges follow ElevenLabs' voice_settings API (stability/similarity/style
    # 0-1, speed roughly 0.7-1.2) - see app/pipeline/tts.py.
    elevenlabs_voice_stability: float = 0.35
    elevenlabs_voice_similarity: float = 0.8
    elevenlabs_voice_style: float = 0.55
    elevenlabs_voice_speed: float = 1.05

    # Below this confidence, a visual-attention effect's target localization
    # isn't trusted enough to draw a circle/arrow pointing at it (a wrong
    # confident guess - an arrow on the wrong car - is worse than no
    # annotation at all). See app/pipeline/geometry.py.
    effects_target_confidence_threshold: float = 0.75

    # Storage
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = "road-rage-clipper"
    r2_endpoint_url: str = ""

    # Local paths (used for scratch work always, and as storage fallback)
    data_dir: str = "./data"

    # Retention / limits
    retention_hours: float = 48
    max_video_mb: float = 500
    max_video_duration_seconds: float = 1800
    download_timeout_seconds: float = 600

    # FFmpeg encode/filter thread cap. Small Railway containers report the
    # *host's* full CPU count (we've seen ffmpeg auto-detect 60 threads on a
    # container with a fraction of that memory), so libx264's default
    # thread-count auto-detection can spin up far more threads than the
    # container can afford and get SIGKILLed by the OOM killer. 2 is a safe
    # default that still parallelizes a bit without blowing up memory.
    ffmpeg_threads: int = 2

    # Misc
    base_url: str = "http://localhost:8000"
    port: int = 8000

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def tmp_path(self) -> Path:
        p = self.data_path / "tmp"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def local_storage_path(self) -> Path:
        p = self.data_path / "storage"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.data_path / "jobs.db"

    @property
    def uses_r2(self) -> bool:
        return bool(self.r2_account_id and self.r2_access_key_id and self.r2_secret_access_key)

    @property
    def resolved_r2_endpoint(self) -> str:
        if self.r2_endpoint_url:
            return self.r2_endpoint_url
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"


@lru_cache
def get_settings() -> Settings:
    # Railway sets PORT at runtime; honor it if present and not overridden.
    settings = Settings()
    env_port = os.environ.get("PORT")
    if env_port and settings.port == 8000:
        try:
            settings.port = int(env_port)
        except ValueError:
            pass
    railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    if railway_domain and settings.base_url == "http://localhost:8000":
        settings.base_url = f"https://{railway_domain}"
    return settings
