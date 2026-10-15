from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]  # .../histopathology_pnw


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PNW_", env_file=ROOT / ".env")

    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"
    outputs_dir: Path = ROOT / "outputs"
