"""
CLI Entrypoint
"""
import typer

from .datasets import app as datasets
from .projects import app as projects

app = typer.Typer()
app.add_typer(datasets, name="dataset")
app.add_typer(projects, name="project")

if __name__ == "__main__":
    app()
