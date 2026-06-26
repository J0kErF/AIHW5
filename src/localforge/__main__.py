"""Enable `python -m localforge` as an alias for the CLI entry point."""

from localforge.cli.app import app

if __name__ == "__main__":
    app()
