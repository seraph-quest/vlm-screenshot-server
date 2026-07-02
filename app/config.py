from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8088
    vlm_base_url: str = "http://127.0.0.1:8000/v1"
    vlm_api_key: str = ""
    vlm_model: str = "google/gemma-4-12b-it-qat-q4"
    vlm_timeout_seconds: int = 120
    vlm_analyze_attempts: int = 2
    vlm_max_tokens: int = 700
    vlm_temperature: float = 0
    redact_visible_text: bool = True
    vlm_trust_env: bool = False  # keep LAN calls off ambient HTTP(S)_PROXY by default
    chat_proxy_enabled: bool = False
    chat_proxy_api_key: str = ""
    queue_max_size: int = 8
    queue_workers: int = 1
    queue_background_workers: int = 1
    queue_admit_timeout_seconds: float = 1.0
    queue_result_timeout_seconds: float = 600.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
