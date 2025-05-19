from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    API_KEY: str
    LLM_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    OMDB_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env")


SETTINGS = Settings()
