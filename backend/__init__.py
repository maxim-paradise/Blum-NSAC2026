"""Configuration and secrets helpers (env vars with .streamlit/secrets.toml fallback)."""

from functools import lru_cache
import os
from pathlib import Path
import tomllib

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "experimental_dataset.json"
SECRETS_PATH = ROOT_DIR / ".streamlit" / "secrets.toml"
FRONTEND_DIST_DIR = ROOT_DIR / "frontend" / "dist"

COARENA_ANTHROPIC_BASE_URL = "https://coarena.ai/api"
COARENA_OPENAI_BASE_URL = "https://coarena.ai/api/v1"
COARENA_MODELS_URL = f"{COARENA_OPENAI_BASE_URL}/models"
COARENA_FEEDBACK_URL = f"{COARENA_OPENAI_BASE_URL}/feedback"

DEFAULT_MODEL = "deepseek-v4-1-flash-1017"
DEFAULT_EFFORT = "medium"
REQUEST_TIMEOUT_SEC = 60.0
MAX_TOKENS = 2400
THINKING_BUDGET = 1200
MAX_AGENT_STEPS = 6


@lru_cache(maxsize=1)
def _load_toml_secrets() -> dict[str, str]:
    if not SECRETS_PATH.is_file():
        return {}
    try:
        with SECRETS_PATH.open("rb") as f:
            data = tomllib.load(f)
        return {str(k): str(v).strip() for k, v in data.items() if v is not None}
    except Exception:
        return {}


def get_setting(name: str, default: str = "") -> str:
    env_val = os.environ.get(name, "").strip()
    if env_val:
        return env_val
    toml_val = _load_toml_secrets().get(name, "").strip()
    return toml_val or default


def get_api_key() -> str:
    return (
        get_setting("COARENA_API_KEY")
        or get_setting("ANTHROPIC_API_KEY")
        or get_setting("OPENAI_API_KEY")
    )


def get_base_url() -> str:
    return get_setting("COARENA_BASE_URL", COARENA_ANTHROPIC_BASE_URL).rstrip("/")


def get_default_model() -> str:
    return get_setting("COARENA_MODEL", DEFAULT_MODEL)


def get_default_effort() -> str:
    return get_setting("COARENA_EFFORT", DEFAULT_EFFORT)
