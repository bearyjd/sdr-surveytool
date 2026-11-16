"""docker-compose.yml must not ship default DB credentials or expose Postgres publicly."""

from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parent.parent / "docker-compose.yml"


def _postgres() -> dict:
    return yaml.safe_load(COMPOSE.read_text())["services"]["postgres"]


def test_password_required_from_env_without_default():
    password = _postgres()["environment"]["POSTGRES_PASSWORD"]
    # ${VAR:?msg} fails compose up when unset; ${VAR:-x} / literals are defaults.
    assert password.startswith("${POSTGRES_PASSWORD:?"), password


def test_port_bound_to_loopback_only():
    for port in _postgres()["ports"]:
        assert str(port).startswith("127.0.0.1:"), port


def test_dotenv_is_gitignored_and_example_shipped():
    root = COMPOSE.parent
    assert ".env" in (root / ".gitignore").read_text().splitlines()
    assert (root / ".env.example").read_text().startswith("POSTGRES_PASSWORD=")
