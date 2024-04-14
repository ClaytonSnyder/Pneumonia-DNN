import os
import time

from typing import Any

import keras
import numpy as np
import tensorflow as tf

from keras import Input, layers
from vit_keras import vit

from pneumonia_dnn.utils import (
    get_augmented_inputs,
    get_project_configuration,
    get_project_datasets,
    save_train_session,
)


def create_model(
    project_name: str,
    activation: str,
    pretrained: bool,
    include_top: bool,
    pretrained_top: bool,
    weights: str,
    projects_path: str,
):
    project_data = get_project_configuration(project_name, projects_path)

    height = project_data["image_height"]
    width = project_data["image_width"]
    num_classes = len(project_data["labels"])
    channels = project_data["image_channels"]

    augmentation = get_augmented_inputs(project_name, projects_path)

    inputs = Input(shape=(height, width, channels))
    augmented = augmentation(inputs)

    pre_trained_model = vit.vit_b16(
        image_size=(height, width),
        activation=activation,
        pretrained=pretrained,
        include_top=include_top,
        pretrained_top=pretrained_top,
        weights=weights,
        classes=num_classes,
    )

    for layer in pre_trained_model.layers:
        layer.trainable = False

    pretrained_outputs = pre_trained_model(augmented)

    # Classify outputs.
    if num_classes == 2:
        logits = layers.Dense(1, activation="sigmoid")(pretrained_outputs)
    else:
        logits = layers.Dense(num_classes, activation="softmax")(pretrained_outputs)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def run_vitb16(
    project_name: str,
    activation="sigmoid",
    pretrained=True,
    include_top=False,
    pretrained_top=False,
    weights="imagenet21k+imagenet2012",
    learning_rate: float = 0.0001,
    epochs: int = 25,
    batch_size: int = 32,
    projects_path: str = "projects",
) -> Any:
    """
    Run Model

    Args:
        project_name: _description_
        batch_size: _description_. Defaults to 32.
        patch_size: _description_. Defaults to 6.
        projection_dim: _description_. Defaults to 32.
        num_heads: _description_. Defaults to 2.
        transformer_layers: _description_. Defaults to 3.
        learning_rate: Learning rate of loss function
        epochs: Number of epochs
        batch_size: Batch size
        projects_path: _description_. Defaults to "projects".

    Returns:
        VIT Train Session Results
    """
    keras.mixed_precision.set_global_policy("mixed_float16")
    train_dataset, test_dataset = get_project_datasets(project_name, projects_path)

    model = create_model(
        project_name,
        activation,
        pretrained,
        include_top,
        pretrained_top,
        weights,
        projects_path,
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        batch_size=batch_size,
    )

    save_train_session(
        "vitb16",
        time.strftime("%Y-%m-%dT%H%M%S"),
        project_name,
        model,
        history.history,
        {
            "activation": activation,
            "pretrained": pretrained,
            "include_top": include_top,
            "pretrained_top": pretrained_top,
            "weights": weights,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        projects_path,
    )

    return history
