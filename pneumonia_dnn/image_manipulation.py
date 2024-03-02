"""
Image Manipulation
"""

import json
import os

from typing import Any, Dict, List

import cv2

from PIL import Image
from tqdm import tqdm


def grayscale(project_name: str, projects_path: str) -> None:
    """
     Converts Images to grayscale

    Args:
        project_name (str): Project Name
        projects_path (str): Projects path
    """
    images = _get_all_images(project_name, projects_path)

    for image in tqdm(images, desc="Resizing images..."):
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(image, gray)

    _save_manipulation(project_name, projects_path, {"grayscale": True})


def resize_images(
    project_name: str, projects_path: str, width: int, height: int
) -> None:
    """
     Resize Images for a project

    Args:
        project_name (str): Project Name
        projects_path (str): Projects Path
        width (int): Width
        height (int): Height
    """
    images = _get_all_images(project_name, projects_path)

    for image in tqdm(images, desc="Resizing images..."):
        source = Image.open(image)
        new_image = source.resize((width, height))
        new_image.save(image)

    _save_manipulation(project_name, projects_path, {"width": width, "height": height})


def _get_all_images(project_name: str, projects_path: str) -> List[str]:
    project_path = f"{projects_path}/{project_name}"
    test_positive_path = f"{project_path}/dataset/test/pneumonia"
    test_negative_path = f"{project_path}/dataset/test/nonpneumonia"
    train_postive_path = f"{project_path}/dataset/train/pneumonia"
    train_negative_path = f"{project_path}/dataset/train/nonpneumonia"

    test_positive_images = [
        f"{test_positive_path}/{image}" for image in os.listdir(test_positive_path)
    ]
    test_negative_images = [
        f"{test_negative_path}/{image}" for image in os.listdir(test_negative_path)
    ]
    train_positive_images = [
        f"{train_postive_path}/{image}" for image in os.listdir(train_postive_path)
    ]
    train_negative_images = [
        f"{train_negative_path}/{image}" for image in os.listdir(train_negative_path)
    ]

    images = test_positive_images
    images.extend(test_negative_images)
    images.extend(train_positive_images)
    images.extend(train_negative_images)

    return images


def _save_manipulation(
    project_name: str, projects_path: str, data: Dict[Any, Any]
) -> None:
    file_path = f"{projects_path}/{project_name}/project.json"

    with open(file_path, "r", encoding="utf-8") as f:
        project_json = json.load(f)

    project_json["image_manipulations"].append(data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(project_json, f)
