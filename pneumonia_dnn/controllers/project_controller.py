from flask import Blueprint, request

from pneumonia_dnn import project


project_blueprint = Blueprint(name="project_blueprint", import_name=__name__)


@project_blueprint.route("/", methods=["POST"])
def create_dataset():
    data = request.get_json()
    return project.create_project(
        data["name"],
        data["dataset_name"],
        data["image_width"],
        data["image_height"],
        3,
        data["image_split"],
        data["train_split"],
        data["max_images"],
        data["seed"],
    )


@project_blueprint.route("/", methods=["GET"])
def get_projects():
    return project.get_projects()
