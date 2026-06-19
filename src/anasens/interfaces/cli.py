"""Command-line interface for AnaSens2."""

import typer

from anasens.version import __version__

app = typer.Typer(help="AnaSens2 analysis pipeline CLI")


@app.callback()
def main(version: bool = typer.Option(False, "--version", help="Show package version and exit.")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def gui() -> None:
    """Launch package GUI launcher."""
    from anasens.interfaces.gui_app import launch_gui

    launch_gui()


if __name__ == "__main__":
    app()
