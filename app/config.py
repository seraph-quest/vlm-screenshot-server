from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8088
    vlm_base_url: str = "http://127.0.0.1:8000/v1"
    vlm_api_key: str = ""
    vlm_model: str = "google/gemma-4-12b-it-qat-q4"
    vlm_timeout_seconds: int = 120
    vlm_max_tokens: int = 700
    vlm_temperature: float = 0
    redact_visible_text: bool = True
    vlm_trust_env: bool = False  # keep LAN calls off ambient HTTP(S)_PROXY by default

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
