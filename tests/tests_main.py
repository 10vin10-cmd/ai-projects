
from click.testing import CliRunner

from ai_toolbox import main


def test_cli_has_group():
    # ensure the module exposes the click group
    assert hasattr(main, "cli")


def test_hello_command_outputs_greeting():
    runner = CliRunner()
    result = runner.invoke(main.cli, ["hello"])
    assert result.exit_code == 0
    assert "Hello from the AI toolbox!" in result.output
