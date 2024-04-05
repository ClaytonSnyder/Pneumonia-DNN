"""
Preprocessor CLI
"""

from typing import Optional

import typer

from pneumonia_dnn.image_manipulation import resize_images
from pneumonia_dnn.models.cnn import run_model as run_cnn
from pneumonia_dnn.models.vision_transformer import run_model as run_vit
from pneumonia_dnn.preprocessor import create_project, delete_project


app = typer.Typer()


@app.command()
def delete(project_name: str, projects_path: str = "projects"):
    """
    Delete a project

    Args:
        project_name (str): Name of project to delete
        projects_path (_type_, optional): Path where projects will be created.
    """
    delete_project(project_name, projects_path, True)


@app.command()
def resize(  # pylint: disable=too-many-arguments
    name: str,
    width: int,
    height: int,
    projects_path: str = "projects",
):
    """
    _summary_

    Args:
        name (str): Name of the project
        width (int): Width to scale images to.
        height (int): Height to scale images to.
        projects_path (str, optional): Path to projects folder.
    """
    resize_images(name, projects_path, width, height)


@app.command()
def create(  # pylint: disable=too-many-arguments
    name: str,
    max_images: int = 100,
    percent_training: float = 0.7,
    percent_pneumonia: float = 0.5,
    datasets_path: str = "datasets",
    projects_path: str = "projects",
    seed: Optional[int] = None,
):
    """
    Creates a project

    Args:
        name (str): Name of the project
        max_images (int, optional): Maximum number of images in the dataset.
        percent_training (float, optional): Percentage of images to be used for training.
        percent_pneumonia (float, optional): Percentage of images that are labelled as pneumonia.
        datasets_path (str, optional): Path to downloaded datasets.
        projects_path (_type_, optional): Path where projects will be created.
        seed (int, optional): Seed for project recreation
    """
    create_project(
        name,
        max_images,
        percent_training,
        percent_pneumonia,
        datasets_path,
        projects_path,
        seed,
    )


@app.command()
def cnn(  # pylint: disable=too-many-arguments
    name: str, width: int = 500, height: int = 500, projects_path: str = "projects"
):
    """
    Creates a project

    Args:
        name (str): Name of the project
        width (int): Width to scale images to.
        height (int): Height to scale images to.
        projects_path (_type_, optional): Path where projects will be created.
    """
    run_cnn(name, projects_path, width, height, 3)


@app.command()
def vit(  # pylint: disable=too-many-arguments
    name: str,
    width: int = 500,
    height: int = 500,
    projects_path: str = "projects",
    patch_size=10,
    num_layers=6,
    d_model=256,
    num_heads=8,
    mlp_dim=512,
    dropout_rate=0.1,
):
    """
    Creates a project

    Args:
        name (str): Name of the project
        width (int): Width to scale images to.
        height (int): Height to scale images to.
        projects_path (_type_, optional): Path where projects will be created.
    """
    run_vit(
        name,
        projects_path,
        width,
        height,
        3,
        patch_size,
        num_layers,
        d_model,
        num_heads,
        mlp_dim,
        dropout_rate,
    )


if __name__ == "__main__":
    app()
