"""
Preprocessor Class
"""

import os

from dataclasses import dataclass
from shutil import copyfile
from typing import List, Optional

import pandas as pd

from kaggle.api.kaggle_api_extended import KaggleApi


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

            if folder_column is not None:
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