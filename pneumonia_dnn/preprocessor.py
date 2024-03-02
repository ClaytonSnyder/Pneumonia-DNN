"""
Preprocessor Class
"""

import json
import os
import random
import shutil

from math import ceil
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

from pneumonia_dnn.dataset_processors.base_processor import (
    DatasetProcessorBase,
    ProcessedImage,
)
from pneumonia_dnn.dataset_processors.coronahack_processor import CoronahackProcessor
from pneumonia_dnn.dataset_processors.nih_processor import NIHSampleProcessor


class PreprocessingError(Exception):
    """
    Preprocessing Errors

    Args:
        Exception (_type_): Exception
    """

    def __init__(self, message):
        super().__init__(message)


PROCESSORS: List[DatasetProcessorBase] = [CoronahackProcessor(), NIHSampleProcessor()]


def download_datasets(datasets_path: str):
    """
    Download Datasets
    """
    api = KaggleApi()
    api.authenticate()

    Path(datasets_path).mkdir(parents=True, exist_ok=True)

    for processor in PROCESSORS:
        dataset_path = processor.get_output_path(datasets_path)
        identifier = processor.get_dataset_identifier()
        if not os.path.exists(dataset_path):
            api.dataset_download_files(
                identifier, path=dataset_path, unzip=True, quiet=False
            )
        else:
            print(
                f"{identifier} already exists. Please execute pdnn datasets delete {dataset_path}"
            )


def delete_datasets(datasets_path: str, prompt_confirmation: bool):
    """
    Delete downloaded datasets

    Args:
        datasets_path (str): Path to datasets
        prompt_confirmation (bool): If true user input confirmation is required
    """
    _delete_path(datasets_path, prompt_confirmation)


def delete_project(name: str, projects_path: str, prompt_confirmation: bool):
    """
    Delete a project

    Args:
        name (str): Delete a project
        projects_path (str): Projects path
        prompt_confirmation (bool): If true user input confirmation is required
    """
    project_path = f"{projects_path}/{name}"
    _delete_path(project_path, prompt_confirmation)


def _delete_path(path: str, prompt_confirmation: bool):
    should_delete = True

    if prompt_confirmation:
        answer = ""
        while answer not in ["y", "n"]:
            answer = input(f"Are you sure you want to delete {path} [Y/N]? ").lower()
        should_delete = answer == "y"

    if should_delete:
        if not os.path.exists(path):
            raise PreprocessingError(f"Cannot delete {path} because it doesnt exist.")

        shutil.rmtree(path)


def create_project(
    name: str,
    max_images: int,
    percent_training: float,
    percent_pneumonia: float,
    datasets_path: str,
    projects_path: str,
    seed: Optional[int],
):
    """
    Preprocess images

    Args:
        max_images (int): Max number of images in filtered dataset
    """

    if seed is None:
        seed = random.randint(0, 10)
    np.random.seed(1)

    download_datasets(datasets_path)
    project_path = f"{projects_path}/{name}/dataset/"
    train_path = f"{project_path}train"
    test_path = f"{project_path}test"

    if os.path.exists(project_path):
        raise PreprocessingError(f"Dataset {name} already exists.")

    if percent_pneumonia > 1 or percent_pneumonia < 0:
        raise PreprocessingError("percent_pneumonia must be between 0 and 1")

    if percent_training > 1 or percent_training < 0:
        raise PreprocessingError("percent_training must be between 0 and 1")

    Path(project_path).mkdir(parents=True, exist_ok=True)
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(f"{train_path}/pneumonia").mkdir(parents=True, exist_ok=True)
    Path(f"{train_path}/nonpneumonia").mkdir(parents=True, exist_ok=True)
    Path(test_path).mkdir(parents=True, exist_ok=True)
    Path(f"{test_path}/pneumonia").mkdir(parents=True, exist_ok=True)
    Path(f"{test_path}/nonpneumonia").mkdir(parents=True, exist_ok=True)

    max_processor_images = ceil(max_images / len(PROCESSORS))

    num_of_train: int = ceil(max_processor_images * percent_training)
    num_of_test: int = max_processor_images - num_of_train

    num_of_pneumonia_train: int = ceil(num_of_train * percent_pneumonia)
    num_of_nonpneumonia_train: int = num_of_train - num_of_pneumonia_train
    num_of_pneumonia_test: int = ceil(num_of_test * percent_pneumonia)
    num_of_nonpneumonia_test: int = num_of_test - num_of_pneumonia_test

    images: List[ProcessedImage] = []

    for processor in tqdm(PROCESSORS, desc="Collecting dataset images..."):
        images.extend(
            processor.get_images(
                datasets_path,
                num_of_pneumonia_train,
                num_of_nonpneumonia_train,
                num_of_pneumonia_test,
                num_of_nonpneumonia_test,
            )
        )

    image_paths: List[str] = []
    has_pneumonia: List[bool] = []
    label: List[str] = []
    is_training: List[bool] = []
    original_filepath: List[str] = []

    for image in tqdm(images, desc="Processing images..."):
        if image.is_training:
            destination_path = train_path
        else:
            destination_path = test_path

        if image.has_pneumonia:
            destination_path += "/pneumonia"
        else:
            destination_path += "/nonpneumonia"

        destination_path += f"/{os.path.basename(image.image_path)}"

        shutil.copy(image.image_path, destination_path)

        image_paths.append(destination_path)
        has_pneumonia.append(image.has_pneumonia)
        label.append(image.label.name)
        is_training.append(image.is_training)
        original_filepath.append(image.image_path)

    project_data = {
        "name": name,
        "max_images": max_images,
        "percent_training": percent_training,
        "percent_pneumonia": percent_pneumonia,
        "training_path": train_path,
        "testing_path": test_path,
        "seed": seed,
        "image_manipulations": [],
    }

    with open(f"{projects_path}/{name}/project.json", "w", encoding="utf-8") as fout:
        json.dump(project_data, fout)

    metadata_df = pd.DataFrame(
        {
            "image": image_paths,
            "has_pneumonia": has_pneumonia,
            "label": label,
            "is_training": is_training,
            "original_filepath": original_filepath,
        }
    )

    metadata_df.to_csv(f"{projects_path}/{name}/metadata.csv")
