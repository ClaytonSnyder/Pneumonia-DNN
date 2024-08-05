"""
Preprocessor Class
"""

import glob
import json
import os
import re
import shutil

from dataclasses import dataclass
from math import ceil
from shutil import copyfile
from typing import Any, Dict, List, Optional

import pandas as pd

from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image


class DatasetError(Exception):
    """
    Dataset Errors

    Args:
        Exception (_type_): Exception
    """

    def __init__(self, message):
        super().__init__(message)


def __download(dataset_identifier: str, download_path: str) -> str:
    api = KaggleApi()
    api.authenticate()

    os.makedirs(download_path, exist_ok=True)

    dataset_name = dataset_identifier.split("/")[-1]
    dataset_path = f"{download_path}/{dataset_name}"

    if os.path.exists(dataset_path):
        return dataset_name

    api.dataset_download_files(
        dataset_identifier, path=dataset_path, unzip=True, quiet=False
    )
    return dataset_name


@dataclass
class DatasetClassifier:
    label: str
    folder: str
    alias: Optional[str]
    included: bool = True


def create_update_dataset_from_metadata(
    dataset_name: str,
    dataset_identifier: str,
    path_to_metadata: str,
    label_column: str,
    labels: List[DatasetClassifier],
    image_column: str,
    folder_column: Optional[str] = None,
    folder_to_lower: bool = True,
    download_path: str = "downloads",
    dataset_path: str = "datasets",
):
    downloaded_dataset_name = __download(dataset_identifier, download_path)

    metadata_df = pd.read_csv(
        f"{download_path}/{downloaded_dataset_name}/{path_to_metadata}"
    )

    all_data_location = f"{dataset_path}/{dataset_name}"
    os.makedirs(all_data_location, exist_ok=True)

    all_data_metadata_path = f"{all_data_location}/__metadata.csv"
    if os.path.exists(all_data_metadata_path):
        current_metadata = pd.read_csv(all_data_metadata_path).to_dict("dict")
        files = list(current_metadata["file"].values())
        categories = list(current_metadata["label"].values())
    else:
        files = []
        categories = []

    for label_data in labels:
        if label_data.alias is not None:
            if label_data.included:
                category_df = metadata_df[
                    metadata_df[label_column].str.contains(label_data.alias)
                ]
            else:
                category_df = metadata_df[
                    ~metadata_df[label_column].str.contains(label_data.alias)
                ]

            if folder_column is not None and folder_column != "":
                if folder_to_lower:
                    category_df[image_column] = (
                        label_data.folder
                        + "/"
                        + category_df[folder_column].str.lower()
                        + "/"
                        + category_df[image_column]
                    )
                else:
                    category_df[image_column] = (
                        label_data.folder
                        + "/"
                        + category_df[folder_column]
                        + "/"
                        + category_df[image_column]
                    )
            else:
                category_df[image_column] = (
                    label_data.folder + "/" + category_df[image_column]
                )

            for row in category_df[image_column].tolist():
                destination = f"{all_data_location}/{os.path.basename(row)}"
                copyfile(
                    f"{download_path}/{downloaded_dataset_name}/{row}", destination
                )
                files.append(destination)
                categories.append(label_data.label)

    all_data_df = pd.DataFrame(
        {
            "file": files,
            "label": categories,
        }
    )

    all_data_df.to_csv(f"{all_data_location}/__metadata.csv")


def update_dataset_from_metadata(
    dataset_name: str,
    dataset_identifier: str,
    dataset_kaggle_url: str,
    path_to_metadata: str,
    label_column: str,
    labels: List[DatasetClassifier],
    image_column: str,
    folder_column: Optional[str] = None,
    folder_to_lower: bool = True,
    download_path: str = "downloads",
    dataset_path: str = "datasets",
):
    all_data_location = f"{dataset_path}/{dataset_name}"

    if not os.path.exists(all_data_location):
        raise DatasetError(f"Project {dataset_name} doesnt exist.")

    downloaded_dataset_name = __download(dataset_identifier, download_path)

    metadata_df = pd.read_csv(
        f"{download_path}/{downloaded_dataset_name}/{path_to_metadata}"
    )

    all_data_metadata_path = f"{all_data_location}/__metadata.csv"
    if os.path.exists(all_data_metadata_path):
        current_metadata = pd.read_csv(all_data_metadata_path).to_dict("dict")
        files = list(current_metadata["file"].values())
        categories = list(current_metadata["label"].values())
        all_widths = list(current_metadata["width"].values())
        all_heights = list(current_metadata["height"].values())
    else:
        files = []
        categories = []
        all_widths: List[int] = []
        all_heights: List[int] = []

    label_counts: Dict[str, int] = {}
    widths: Dict[str, int] = {}
    heights: Dict[str, int] = {}
    corrupt_image_count = 0

    for label_data in labels:
        if label_data.alias is not None:
            if label_data.included:
                category_df = metadata_df[
                    metadata_df[label_column].str.contains(label_data.alias)
                ]
            else:
                category_df = metadata_df[
                    ~metadata_df[label_column].str.contains(label_data.alias)
                ]

            if folder_column is not None and folder_column != "":
                if folder_to_lower:
                    category_df[image_column] = (
                        label_data.folder
                        + "/"
                        + category_df[folder_column].str.lower()
                        + "/"
                        + category_df[image_column]
                    )
                else:
                    category_df[image_column] = (
                        label_data.folder
                        + "/"
                        + category_df[folder_column]
                        + "/"
                        + category_df[image_column]
                    )
            else:
                category_df[image_column] = (
                    label_data.folder + "/" + category_df[image_column]
                )

            rows = category_df[image_column].tolist()
            local_corrupt_images = 0
            for row in rows:
                destination = f"{all_data_location}/{os.path.basename(row)}"
                copyfile(
                    f"{download_path}/{downloaded_dataset_name}/{row}", destination
                )

                try:
                    img = Image.open(destination)
                    img.verify()
                    width, height = img.size
                    width_str = str(width)
                    height_str = str(height)

                    if width_str not in widths:
                        widths[width_str] = 0

                    widths[width_str] = widths[width_str] + 1

                    if height_str not in heights:
                        heights[height_str] = 0

                    heights[height_str] = heights[height_str] + 1

                    all_heights.append(height)
                    all_widths.append(width)
                    files.append(destination)
                    categories.append(label_data.label)
                except (IOError, SyntaxError) as e:
                    os.remove(destination)
                    local_corrupt_images = local_corrupt_images + 1
                    corrupt_image_count = local_corrupt_images + 1

            label_counts[label_data.label] = len(rows) - local_corrupt_images

    all_data_df = pd.DataFrame(
        {
            "file": files,
            "label": categories,
            "width": all_widths,
            "height": all_heights,
        }
    )

    # Update the dataset properties file
    properties_file_path = f"{all_data_location}/__properties.json"
    with open(properties_file_path, "r", encoding="utf-8") as properties_file:
        dataset_properties = json.load(properties_file)

    for label in label_counts:
        if label in dataset_properties["labels"]:
            dataset_properties["labels"][label] = (
                dataset_properties["labels"][label] + label_counts[label]
            )
        else:
            dataset_properties["labels"][label] = label_counts[label]

    dataset_properties["total_image_count"] = len(files)
    kaggle_datasets = dataset_properties["kaggle_datasets_included"]
    dataset_properties["corrupt_image_count"] = (
        dataset_properties["corrupt_image_count"] + corrupt_image_count
    )

    current_heights = dataset_properties["height_distribution"]

    for height_str in heights:
        if height_str in current_heights:
            current_heights[height_str] = (
                current_heights[height_str] + heights[height_str]
            )
        else:
            current_heights[height_str] = heights[height_str]

    counter = 0
    sum = 0
    for height, count in current_heights.items():
        counter = counter + count
        sum = sum + (int(height) * count)

    if sum > 0:
        avg_height = ceil(sum / counter)
    else:
        avg_height = 0

    sorted_heights = {}
    for sorted_key in sorted(
        current_heights, key=lambda k: int(k), reverse=False
    ):  # type: ignore
        sorted_heights[sorted_key] = current_heights[sorted_key]
    current_heights = sorted_heights

    current_widths = dataset_properties["width_distribution"]

    for width_str in widths:
        if width_str in current_widths:
            current_widths[width_str] = current_widths[width_str] + widths[width_str]
        else:
            current_widths[width_str] = widths[width_str]

    sorted_widths = {}
    for sorted_key in sorted(
        current_widths, key=lambda k: int(k), reverse=False
    ):  # type: ignore
        sorted_widths[sorted_key] = current_widths[sorted_key]
    current_widths = sorted_widths

    counter = 0
    sum = 0
    for width, count in current_widths.items():
        counter = counter + count
        sum = sum + (int(width) * count)

    if sum > 0:
        avg_width = ceil(sum / counter)
    else:
        avg_width = 0

    dataset_properties["width_distribution"] = current_widths
    dataset_properties["height_distribution"] = current_heights
    dataset_properties["avg_width"] = avg_width
    dataset_properties["avg_height"] = avg_height

    kaggle_datasets.append(
        {
            "identifier": dataset_identifier,
            "url": dataset_kaggle_url,
            "path_to_metadata": path_to_metadata,
            "label_column": label_column,
            "image_column": image_column,
            "folder_column": folder_column,
            "folder_to_lower": folder_to_lower,
            "labels": [
                {
                    "label": classifier.label,
                    "folder": classifier.folder,
                    "alias": classifier.alias,
                    "included": classifier.included,
                }
                for classifier in labels
            ],
        }
    )

    with open(properties_file_path, "w", encoding="utf-8") as properties_file:
        json.dump(dataset_properties, properties_file)

    # Save the metadata CSV
    all_data_df.to_csv(f"{all_data_location}/__metadata.csv")


def create_dataset(name: str, description: str, dataset_path: str = "datasets"):
    all_data_location = f"{dataset_path}/{name}"
    if os.path.exists(all_data_location):
        raise DatasetError(f"Project {name} already exists.")

    os.makedirs(all_data_location, exist_ok=True)

    dataset_info: Dict[str, Any] = {
        "name": name,
        "description": description,
        "labels": {},
        "width_distribution": {},
        "height_distribution": {},
        "avg_height": 0,
        "avg_width": 0,
        "total_image_count": 0,
        "corrupt_image_count": 0,
        "kaggle_datasets_included": [],
    }

    with open(
        f"{all_data_location}/__properties.json", "w", encoding="utf-8"
    ) as properties_file:
        json.dump(dataset_info, properties_file)

    return dataset_info


def get_datasets(dataset_path: str = "datasets"):
    dataset_folders = os.listdir(dataset_path)

    datasets = []

    for set_path in dataset_folders:
        properties_path = f"{dataset_path}/{set_path}/__properties.json"
        with open(properties_path, "r", encoding="utf-8") as properties_file:
            dataset_properties = json.load(properties_file)
            datasets.append(dataset_properties)

    return {"datasets": datasets}


def download_dataset(dataset_identifier: str, download_path: str = "downloads"):

    downloaded_dataset_name = __download(dataset_identifier, download_path)

    downloaded_path = f"{download_path}/{downloaded_dataset_name}/"
    csv_files = []
    subfolders = [""]

    for root, dirs, files in os.walk(downloaded_path):
        for dir in dirs:
            full_dir_path = os.path.join(root, dir)
            relative_path = os.path.relpath(full_dir_path, start=downloaded_path)
            subfolders.append(relative_path)

        pattern = os.path.join(root, "*.csv")
        for csv_file in glob.glob(pattern):
            relative_path = os.path.relpath(csv_file, start=downloaded_path)
            csv_files.append(relative_path)

    return {"csv": csv_files, "subfolders": subfolders}


def get_columns(
    dataset_identifier: str, path_to_metadata: str, download_path: str = "downloads"
):

    downloaded_dataset_name = __download(dataset_identifier, download_path)

    metadata_df = pd.read_csv(
        f"{download_path}/{downloaded_dataset_name}/{path_to_metadata}"
    )

    return {"columns": list(metadata_df.columns.values)}


def get_column_unique_values(
    dataset_identifier: str,
    path_to_metadata: str,
    column: str,
    download_path: str = "downloads",
):
    downloaded_dataset_name = __download(dataset_identifier, download_path)

    metadata_df = pd.read_csv(
        f"{download_path}/{downloaded_dataset_name}/{path_to_metadata}"
    )

    all_unique = []
    unique_values = metadata_df[column].unique().tolist()

    for unique_value in unique_values:
        if isinstance(unique_value, str):
            pattern = r"\s*,\s*|\s*,|,\s*|\s*,|\|\s*|\s*\||\|\s*"
            instance_unique_values = re.split(pattern, unique_value)
            all_unique.extend(instance_unique_values)

    all_unique = list(set(all_unique))
    all_unique.sort()
    return all_unique


def search_datasets(query: str):
    api = KaggleApi()
    api.authenticate()

    datasets = api.dataset_list(search=query)
    return [
        {
            "description": dataset.description,  # type: ignore
            "creatorName": dataset.creatorName,  # type: ignore
            "lastUpdated": dataset.lastUpdated,  # type: ignore
            "licenseName": dataset.licenseName,  # type: ignore
            "ownerName": dataset.ownerName,  # type: ignore
            "identifier": dataset.ref,  # type: ignore
            "subtitle": dataset.subtitle,  # type: ignore
            "url": dataset.url,  # type: ignore
            "usabilityRating": dataset.usabilityRating,  # type: ignore
            "voteCount": dataset.voteCount,  # type: ignore
            "totalBytes": dataset.totalBytes,  # type: ignore
        }
        for dataset in datasets
    ]
