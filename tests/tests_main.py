from click.testing import CliRunner
from ai_toolbox import main


def test_cli_has_group():
    # ensure the module exposes the click group
    assert hasattr(main, "cli")


def test_hello(mocker):
    """Mock litellm completion via pytest-mock and assert stable output."""
    # Build a stable fake response object that matches the shape used in production
    runner = CliRunner()
    mock_completion = mocker.patch("ai_toolbox.main.completion") 
    mock_completion.return_value = [
        mocker.Mock(
            choices=[
                mocker.Mock(
                    delta=mocker.Mock(
                        content="Mocked completion response."
                    )
                )       
            ]
        )
    ]

    stable_text = "Stubbed greeting from mocked litellm"

    # Patch the symbol imported into the module (ai_toolbox.main.completion)
    
 
    result = runner.invoke(main.cli, ["hello"])

    assert result.exit_code == 0
    assert "Mocked completion response." in result.output
