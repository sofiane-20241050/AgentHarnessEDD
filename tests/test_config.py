"""配置分层测试：默认值 < YAML < .env/环境变量 < CLI overrides。"""

import pytest

from ahedd.config import ModelSpec, load_models_config

CLEAN_ENV: dict[str, str] = {}  # 注入空环境，隔离真实 AHEDD_* 变量


def test_agent_required(tmp_path):
    with pytest.raises(ValueError, match="agent"):
        load_models_config(tmp_path / "missing.yaml", dotenv_path=None, env=CLEAN_ENV)


def test_yaml_layer(tmp_path):
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(
        "agent:\n  name: local\n  model: m1\n  base_url: http://a/v1\n", encoding="utf-8"
    )
    roles = load_models_config(yaml_file, dotenv_path=None, env=CLEAN_ENV)
    assert roles.agent.model == "m1"


def test_env_overrides_yaml(tmp_path):
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text("agent:\n  model: m1\n", encoding="utf-8")
    roles = load_models_config(
        yaml_file, dotenv_path=None, env={"AHEDD_AGENT_MODEL": "m2"}
    )
    assert roles.agent.model == "m2"  # 环境变量 > YAML


def test_cli_overrides_env(tmp_path):
    roles = load_models_config(
        tmp_path / "missing.yaml",
        dotenv_path=None,
        env={"AHEDD_AGENT_MODEL": "m2"},
        overrides={"agent": {"model": "m3", "base_url": "http://cli/v1"}},
    )
    assert roles.agent.model == "m3"  # CLI > 环境变量
    assert roles.agent.base_url == "http://cli/v1"


def test_dotenv_layer(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment\nAHEDD_AGENT_MODEL=m4\nAHEDD_AGENT_BASE_URL=http://dotenv/v1\n",
        encoding="utf-8",
    )
    roles = load_models_config(tmp_path / "missing.yaml", dotenv_path=dotenv, env=CLEAN_ENV)
    assert roles.agent.model == "m4"
    assert roles.agent.base_url == "http://dotenv/v1"


def test_dotenv_does_not_override_real_env(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("AHEDD_AGENT_MODEL=from_dotenv\n", encoding="utf-8")
    roles = load_models_config(
        tmp_path / "missing.yaml",
        dotenv_path=dotenv,
        env={"AHEDD_AGENT_MODEL": "from_env"},  # 真实环境变量优先于 .env
    )
    assert roles.agent.model == "from_env"


def test_field_coercion(tmp_path):
    roles = load_models_config(
        tmp_path / "missing.yaml",
        dotenv_path=None,
        env={
            "AHEDD_AGENT_MODEL": "m",
            "AHEDD_AGENT_TEMPERATURE": "0.7",
            "AHEDD_AGENT_MAX_TOKENS": "4096",
        },
    )
    assert roles.agent.temperature == 0.7
    assert roles.agent.max_tokens == 4096


def test_resolve_api_key(monkeypatch):
    monkeypatch.setenv("MY_KEY_ENV", "k1")
    by_env = ModelSpec(name="a", model="m", api_key_env="MY_KEY_ENV")
    assert by_env.resolve_api_key() == "k1"

    explicit = ModelSpec(name="a", model="m", api_key="k0", api_key_env="MY_KEY_ENV")
    assert explicit.resolve_api_key() == "k0"  # 显式 api_key 优先

    monkeypatch.delenv("MY_KEY_ENV")
    fallback = ModelSpec(name="a", model="m", api_key_env="MY_KEY_ENV")
    assert fallback.resolve_api_key() == "EMPTY"  # vLLM 本地端点
