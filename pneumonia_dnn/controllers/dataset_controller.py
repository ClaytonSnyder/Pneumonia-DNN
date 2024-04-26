"""
Dataset Controller
"""

import datetime

from flask import Blueprint, jsonify, request

from pneumonia_dnn import dataset


# define the blueprint
dataset_blueprint = Blueprint(name="dataset_blueprint", import_name=__name__)


@dataset_blueprint.route("/add", methods=["POST"])  # type: ignore
def add_to_dataset():
    data = request.get_json()

    dataset.update_dataset_from_metadata(
        data["name"],
        data["dataset_identifier"],
        data["dataset_kaggle_url"],
        data["path_to_metadata"],
        data["label_column"],
        [
            dataset.DatasetClassifier(
                label_data["label"],
                label_data["folder"],
                label_data["alias"],
                label_data["included"],
            )
            for label_data in data["labels"]
        ],
        data["image_column"],
        data["folder_column"],
        data["folder_to_lower"],
    )


@dataset_blueprint.route("/download", methods=["POST"])  # type: ignore
def download_dataset():
    data = request.get_json()

    return dataset.download_dataset(
        data["dataset_identifier"],
    )


@dataset_blueprint.route("/columns", methods=["GET"])
def get_columns():
    dataset_identifier = request.args.get("dataset_identifier")
    path_to_metadata = request.args.get("path_to_metadata")
    return dataset.get_columns(dataset_identifier, path_to_metadata)  # type: ignore


@dataset_blueprint.route("/unique", methods=["GET"])
def get_unique_values():
    dataset_identifier = request.args.get("dataset_identifier")
    path_to_metadata = request.args.get("path_to_metadata")
    column = request.args.get("column")
    return {"values": dataset.get_column_unique_values(dataset_identifier, path_to_metadata, column)}  # type: ignore


@dataset_blueprint.route("/kaggle", methods=["GET"])
def find_datasets():
    query = request.args.get("query")
    return dataset.search_datasets(query)  # type: ignore


@dataset_blueprint.route("/", methods=["POST"])
def create_dataset():
    data = request.get_json()
    return dataset.create_dataset(data["name"], data["description"])


@dataset_blueprint.route("/", methods=["GET"])
def get_datasets():
    return dataset.get_datasets()
