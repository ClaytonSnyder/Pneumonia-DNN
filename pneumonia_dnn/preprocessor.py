"""
Preprocessor Class
"""

from math import ceil
import os
from pathlib import Path
from typing import Any, Dict, List
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
import pandas as pd

from pneumonia_dnn.dataset_processors.base_processor import DatasetProcessorBase, ProcessedImage
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

DATASETS: Dict[str,Any] = {
    "paultimothymooney/chest-xray-pneumonia": {
        "output_folder": "paultimothymooney"
    },
    "praveengovi/coronahack-chest-xraydataset": {
        "output_folder": "coronahack"
    },
    "tolgadincer/labeled-chest-xray-images": {
    },
}

PROCESSORS: List[DatasetProcessorBase] = [
    CoronahackProcessor(),
    NIHSampleProcessor()
]

def download_datasets(output_path: str):
    """
    Download Datasets
    """
    api = KaggleApi()
    api.authenticate()

    Path(output_path).mkdir(parents=True, exist_ok=True)

    for processor in PROCESSORS:
        dataset_path = processor.get_output_path(output_path)
        identifier = processor.get_dataset_identifier()
        if not os.path.exists(dataset_path):
            api.dataset_download_files(
                identifier,
                path=dataset_path, unzip=True, quiet=False)
        else:
            print(f"{identifier} already exists. Please execute pdnn datasets delete {dataset_path}")

def delete_datasets():
    """
    Delete existing datasets
    """
    pass

def create_dataset(name: str,
                   max_images: int,
                   percent_training: float,
                   percent_pneumonia: float,
                   output_path: str,
                   width: int,
                   height: int):
    """
    Preprocess images

    Args:
        max_images (int): Max number of images in filtered dataset
    """
    download_datasets(output_path)
    dataset_path = f"{output_path}/{name}"
    train_path = f"{dataset_path}/train"
    test_path = f"{dataset_path}/test"

    if os.path.exists(dataset_path):
        raise PreprocessingError(f"Dataset {name} already exists.")

    if percent_pneumonia > 1 or percent_pneumonia < 0:
        raise PreprocessingError("percent_pneumonia must be between 0 and 1")

    if percent_training > 1 or percent_training < 0:
        raise PreprocessingError("percent_training must be between 0 and 1")

    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(test_path).mkdir(parents=True, exist_ok=True)

    max_processor_images = ceil(max_images/len(PROCESSORS))

    num_of_train: int = ceil(max_processor_images * percent_training)
    num_of_test: int = max_processor_images - num_of_train

    num_of_pneumonia_train: int = ceil(num_of_train * percent_pneumonia)
    num_of_nonpneumonia_train: int = num_of_train - num_of_pneumonia_train
    num_of_pneumonia_test: int = ceil(num_of_test * percent_pneumonia)
    num_of_nonpneumonia_test: int = num_of_test - num_of_pneumonia_test

    images: List[ProcessedImage] = []

    for processor in PROCESSORS:
        images.extend(processor.get_images(
                            output_path,
                            num_of_pneumonia_train,
                             num_of_nonpneumonia_train,
                             num_of_pneumonia_test,
                             num_of_nonpneumonia_test))

    image_paths: List[str] = []
    has_pneumonia: List[bool] = []
    label: List[str] = []
    is_training: List[bool] = []
    original_filepath: List[str] = []

    for image in images:
        if image.is_training:
            destination_path = f"{train_path}/{os.path.basename(image.image_path)}"
        else:
            destination_path = f"{test_path}/{os.path.basename(image.image_path)}"

        source = Image.open(image.image_path)
        new_image = source.resize((width, height))
        new_image.save(destination_path)

        image_paths.append(destination_path)
        has_pneumonia.append(image.has_pneumonia)
        label.append(image.label.name)
        is_training.append(image.is_training)
        original_filepath.append(image.image_path)

    metadata_df = pd.DataFrame({
        "image": image_paths,
        "has_pneumonia": has_pneumonia,
        "label": label,
        "is_training": is_training,
        "original_filepath": original_filepath
    })

    metadata_df.to_csv(f"{output_path}/{name}/metadata.csv")
