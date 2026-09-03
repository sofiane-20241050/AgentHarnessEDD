"""冒烟测试：包可导入、CLI 可用、注册机制工作正常。"""

from click.testing import CliRunner

import ahedd
from ahedd.adapters import list_adapters
from ahedd.cli import main
from ahedd.datasets import list_datasets


def test_version() -> None:
    assert ahedd.__version__ == "0.1.0"


def test_cli_help() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "report", "freeze", "ci", "datasets", "adapters"):
        assert cmd in result.output


def test_cli_lists() -> None:
    assert CliRunner().invoke(main, ["datasets"]).exit_code == 0
    assert CliRunner().invoke(main, ["adapters"]).exit_code == 0


def test_builtin_adapters_registered() -> None:
    names = list_adapters()
    assert "openai-loop" in names
    assert "deepagents" in names
    assert "tau" in names


def test_datasets_registry_empty_ok() -> None:
    # 首个数据集 vita 尚未接入：注册表为空是当前预期状态
    assert isinstance(list_datasets(), list)
