"""
CLI Entrypoint
"""
import typer

from .datasets import app as datasets

app = typer.Typer()
app.add_typer(datasets, name="datasets")

if __name__ == "__main__":
    app()
