"""
Dataset Controller
"""

import datetime

from flask import Blueprint, jsonify, request

from pneumonia_dnn import dataset


# define the blueprint
dataset_blueprint = Blueprint(name="dataset_blueprint", import_name=__name__)


@dataset_blueprint.route("/", methods=["POST"])
def create_dataset():
    data = request.get_json()
    return dataset.create_dataset(data["name"], data["description"])


@dataset_blueprint.route("/", methods=["GET"])
def get_datasets():
    return dataset.get_datasets()
