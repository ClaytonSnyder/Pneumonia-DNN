import json
import os
import random
import shutil

from math import ceil
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pandas import DataFrame
from tqdm import tqdm

from pneumonia_dnn.utils import get_project_configuration


class ProjectError(Exception):
    """
    Preprocessing Errors

    Args:
        Exception (_type_): Exception
    """

    def __init__(self, message):
        super().__init__(message)


def _copy_images(
    df: pd.DataFrame,
    path: str,
    label: str,
    projects_path: str,
    project_name: str,
    height: int,
    width: int,
) -> None:
    df_list = df.values.tolist()

    destination_path = f"{projects_path}/{project_name}/dataset/{path}"

    for row in tqdm(df_list, desc=f"Processing {path} for label {label}"):
        file_path = row[1]
        label_str = row[2]

        os.makedirs(f"{destination_path}/{label_str}/", exist_ok=True)
        shutil.copyfile(
            file_path, f"{destination_path}/{label_str}/{os.path.basename(file_path)}"
        )


def create_project(
    name: str,
    dataset_name: str,
    image_width: int,
    image_height: int,
    image_channels: int,
    label_split: Optional[Dict[str, float]],
    train_split: float,
    max_images: Optional[int],
    seed: Optional[int],
    dataset_path: str = "datasets",
    projects_path: str = "projects",
) -> None:
    """
    Creates a project

    Args:
        name: Name of the project
        target_dataset_path: Dataset Path
        seed: (Optional) Seed
    """

    if not os.path.exists(f"{dataset_path}/{dataset_name}"):
        raise ProjectError(f"{dataset_name} does not exist.")

    if os.path.exists(f"{projects_path}/{name}"):
        raise ProjectError(f"Project with name {name} already exists.")

    os.makedirs(f"{projects_path}/{name}")

    if seed is None:
        seed = random.randint(0, 10)

    np.random.seed(1)

    metadata_df = pd.read_csv(f"{dataset_path}/{dataset_name}/__metadata.csv")
    num_images: int = metadata_df.shape[0]
    labels: List[str] = metadata_df["label"].unique().tolist()
    label_images_lookup: Dict[str, DataFrame] = {}
    train_label_image_count_lookup: Dict[str, int] = {}
    test_label_image_count_lookup: Dict[str, int] = {}

    if max_images is None:
        max_images = num_images

    num_of_train: int = ceil(max_images * train_split)
    num_of_test: int = max_images - num_of_train

    for label in labels:
        label_images_lookup[label] = metadata_df[metadata_df["label"] == label]

        if label_split is None or label not in label_split:
            label_split_percent = 1.0 / len(labels)

        train_label_image_count_lookup[label] = ceil(num_of_train * label_split_percent)
        test_label_image_count_lookup[label] = ceil(num_of_test * label_split_percent)

        if (
            train_label_image_count_lookup[label] + test_label_image_count_lookup[label]
            > label_images_lookup[label].shape[0]
        ):
            raise ProjectError(
                f"Not enough images labeled {label} to split {label_split_percent} of {max_images}."
                + "Decrease the number of max images or decrease the split amount for this label."
            )

        train_data = label_images_lookup[label].sample(
            train_label_image_count_lookup[label]
        )
        _copy_images(
            train_data, "train", label, projects_path, name, image_height, image_width
        )

        test_data = label_images_lookup[label].sample(
            test_label_image_count_lookup[label]
        )
        _copy_images(
            test_data, "test", label, projects_path, name, image_height, image_width
        )

    project_data = {
        "name": name,
        "max_images": max_images,
        "total_train": num_of_train,
        "total_test": num_of_test,
        "train_counts": train_label_image_count_lookup,
        "test_counts": test_label_image_count_lookup,
        "training_path": f"{projects_path}/{name}/dataset/train",
        "testing_path": f"{projects_path}/{name}/dataset/test",
        "seed": seed,
        "image_width": image_width,
        "image_height": image_height,
        "image_channels": image_channels,
        "labels": labels,
    }

    with open(f"{projects_path}/{name}/project.json", "w", encoding="utf-8") as fout:
        json.dump(project_data, fout)


def apply_augmentations(
    project_name: str,
    zoom_height_factor: Optional[float] = None,
    zoom_width_factor: Optional[float] = None,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    random_rotation_factor: Optional[float] = None,
    projects_path: str = "projects",
):
    """
    Adds Augmentations to the current project

    Args:
        project_name: Name of the project
        zoom_height_factor: Random zoom height factor to apply to images. Defaults to None.
        zoom_width_factor: Random zoom width factor to apply to the images. Defaults to None.
        flip_horizontal: When true images will be randomly flipped horizontal. Defaults to False.
        flip_vertical: When true images will be randomly flipped vertically. Defaults to False.
        random_rotation_factor: Random rotation factor. Defaults to None.
        projects_path: _description_. Defaults to "projects".
    """

    project_data = get_project_configuration(project_name, projects_path)

    if "augmentations" not in project_data:
        project_data["augmentations"] = {}

    if zoom_height_factor is not None:
        project_data["augmentations"]["zoom_height_factor"] = zoom_height_factor

    if zoom_width_factor is not None:
        project_data["augmentations"]["zoom_width_factor"] = zoom_width_factor

    if flip_horizontal is not None:
        project_data["augmentations"]["flip_horizontal"] = flip_horizontal

    if flip_vertical is not None:
        project_data["augmentations"]["flip_vertical"] = flip_vertical

    if random_rotation_factor is not None:
        project_data["augmentations"]["random_rotation_factor"] = random_rotation_factor

    with open(
        f"{projects_path}/{project_name}/project.json", "w", encoding="utf-8"
    ) as fout:
        json.dump(project_data, fout)
