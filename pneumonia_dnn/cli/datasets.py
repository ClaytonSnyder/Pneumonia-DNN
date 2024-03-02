"""
Preprocessor CLI
"""
import typer

from pneumonia_dnn.preprocessor import delete_datasets, download_datasets

app = typer.Typer()

@app.command()
def download(datasets_path: str = "datasets"):
    """
     Download Required Datasets

    Args:
        datasets_path (str, optional): Path to downloaded datasets.
    """
    download_datasets(datasets_path)


@app.command()
def delete(datasets_path: str = "datasets"):
    """
    Delete existing datasets

    Args:
        datasets_path (str, optional): Path to downloaded datasets.
    """
    delete_datasets(datasets_path, True)

if __name__ == "__main__":
    app()
