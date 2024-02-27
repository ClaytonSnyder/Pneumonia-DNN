"""
Preprocessor CLI
"""
import typer

from pneumonia_dnn.preprocessor import create_dataset, delete_datasets, download_datasets

app = typer.Typer()

@app.command()
def download(output_path: str = "datasets"):
    """
    Download Required Datasets
    """
    download_datasets(output_path)


@app.command()
def delete():
    """
    Delete existing datasets
    """
    delete_datasets()


@app.command()
def preprocess(name: str,
                max_images: int = 100,
                percent_training: float = 0.7,
                percent_pneumonia: float = 0.5,
                output_path: str = "datasets",
                width: int = 512,
                height: int = 512):
    """
    Preprocesses images and generates a random dataset of max images

    Args:
        max_images (int): _description_
    """
    create_dataset(name, max_images, percent_training,
                   percent_pneumonia, output_path, width, height)

if __name__ == "__main__":
    app()
