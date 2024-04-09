"""
Utilities
"""

import json
import os

from typing import Any, Dict, Optional, Tuple

import keras
import tensorflow as tf

from keras import layers
from keras.utils import image_dataset_from_directory


class DataAugmentationError(Exception):
    """
    Data Augmentation Error

    Args:
        Exception (_type_): Exception
    """

    def __init__(self, message):
        super().__init__(message)


def get_project_configuration(
    project_name: str, projects_folder: str = "projects"
) -> Dict[str, Any]:
    """
    Gets the current project configuration

    Args:
        project_name: _description_
        projects_folder: _description_. Defaults to "projects".

    Returns:
        Dictionary of the project configuration
    """
    project_path = f"{projects_folder}/{project_name}/project.json"

    if not os.path.exists(project_path):
        raise DataAugmentationError(f"Project {project_name} doesn't exist.")

    with open(project_path, "r", encoding="utf-8") as project_file:
        project_data = json.load(project_file)

    return project_data


def get_augmented_inputs(project_name: str, projects_folder: str = "projects") -> Any:
    """
    Preprocessing Network that applies data augmentation

    Args:
        project_name: Name of project
        projects_folder: Projects Path. Defaults to "projects".
    """

    project_data = get_project_configuration(project_name, projects_folder)

    height = project_data["image_height"]
    width = project_data["image_width"]

    augmentations = [layers.Normalization(), layers.Resizing(height, width)]

    if "augmentations" in project_data:
        zoom_height_factor: Optional[float] = project_data["augmentations"].get(
            "zoom_height_factor", None
        )
        zoom_width_factor: Optional[float] = project_data["augmentations"].get(
            "zoom_width_factor", None
        )
        flip_horizontal: bool = project_data["augmentations"].get(
            "flip_horizontal", False
        )
        flip_vertical: bool = project_data["augmentations"].get("flip_vertical", False)
        random_rotation_factor: Optional[float] = project_data["augmentations"].get(
            "random_rotation_factor", None
        )

        if zoom_height_factor is None and zoom_width_factor is not None:
            raise DataAugmentationError(
                "If zoom width factor was defined in augmentation,"
                + " zoom height factor must be as well"
            )

        if flip_horizontal:
            augmentations.append(layers.RandomFlip("horizontal"))

        if flip_vertical:
            augmentations.append(layers.RandomFlip("vertical"))

        if random_rotation_factor is not None:
            augmentations.append(layers.RandomRotation(factor=random_rotation_factor))

        if zoom_height_factor is not None:
            if zoom_width_factor is None:
                augmentations.append(
                    layers.RandomZoom(height_factor=zoom_height_factor)
                )
            else:
                augmentations.append(
                    layers.RandomZoom(
                        height_factor=zoom_height_factor, width_factor=zoom_width_factor
                    )
                )

    return keras.Sequential(
        augmentations,
        name="data_augmentation",
    )


def get_project_datasets(
    project_name: str, projects_path: str = "projects"
) -> Tuple[Any, Any]:
    """
    Gets the train and test data for a project

    Args:
        project_name: Name of project
        projects_path: Projects Path. Defaults to "projects".

    Returns:
        _description_
    """
    train_data_path = f"{projects_path}/{project_name}/dataset/train"
    test_data_path = f"{projects_path}/{project_name}/dataset/test"

    project_data = get_project_configuration(project_name, projects_path)

    height = project_data["image_height"]
    width = project_data["image_width"]
    seed = project_data["seed"]
    labels = project_data["labels"]

    if len(labels) == 2:
        label_mode = "binary"
    else:
        label_mode = "categorical"

    train_dataset = image_dataset_from_directory(
        train_data_path,
        seed=seed,
        image_size=(height, width),
        label_mode=label_mode,
    )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # type: ignore

    test_dataset = image_dataset_from_directory(
        test_data_path,
        seed=seed,
        image_size=(height, width),
        label_mode=label_mode,
    )

    # Configure the dataset for performance
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # type: ignore

    return train_dataset, test_dataset


def save_train_session(
    model_type: str,
    run_name: str,
    project_name: str,
    model: Any,
    history: Dict[Any, Any],
    parameters: Dict[str, Any],
    projects_path: str = "projects",
) -> None:
    """
    Saves a training session

    Args:
        run_name: Name of the run
        project_name: Name of the project
        model: NN Model
        projects_path: Path to projects. Defaults to "projects".
    """
    save_path = f"{projects_path}/{project_name}/train_sessions/{model_type}/{run_name}"
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/model.keras")

    with open(
        f"{save_path}/parameters.json", mode="w", encoding="utf-8"
    ) as parameters_file:
        json.dump(parameters, parameters_file)

    with open(f"{save_path}/history.json", mode="w", encoding="utf-8") as history_file:
        json.dump(history, history_file)
