"""
Convolutional Neural Network
"""

from typing import Any, List, Tuple

import tensorflow as tf

from keras import Input, Model, layers, models
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

from pneumonia_dnn.utils import (
    get_augmented_inputs,
    get_project_configuration,
    get_project_datasets,
)


def __create_model(
    project_name: str,
    kernel_size: Tuple[int, int],
    max_pooling_size: int,
    max_pooling_stride: int,
    cnn_neurons: List[int],
    dropout_rate: float,
    dense_layer_neurons: int,
    projects_path: str,
):
    """
    Create Convolutional Neural Network

    Args:
        project_name: Name of project
        kernel_size: Size of the kernel (Tuple[int, int])
        max_pooling_size: Pooling size of max pooling layers
        max_pooling_stried: Pooling stride of max pooling layers
        cnn_neurons: Number of neurons at each cnn layer
        dropout_rate: Dropout rate
        dense_layer_neurons: Number of neurons at the fully connected dense layer
        projects_path: Projects Path. Defaults to "projects".

    Returns:
        CNN Model
    """
    project_data = get_project_configuration(project_name, projects_path)

    height = project_data["image_height"]
    width = project_data["image_width"]
    channels = project_data["image_channels"]
    augmentation = get_augmented_inputs(project_name, projects_path)
    num_classes = len(project_data["labels"])

    inputs = Input(shape=(height, width, 3))
    augmented = augmentation(inputs)

    initial_conv_layer = layers.Conv2D(
        cnn_neurons[0],
        kernel_size,
        activation="relu",
        input_shape=(width, height, channels),
    )(augmented)

    previous_layer = initial_conv_layer

    for neurons in cnn_neurons[1:]:
        previous_layer = layers.MaxPooling2D(max_pooling_size, max_pooling_stride)(
            previous_layer
        )
        previous_layer = layers.Conv2D(neurons, kernel_size, activation="relu")(
            previous_layer
        )

    flatten_layer = layers.Flatten()(previous_layer)
    dense_layer = layers.Dense(dense_layer_neurons, activation="relu")(flatten_layer)
    dropout_layer = layers.Dropout(dropout_rate)(dense_layer)

    if num_classes == 2:
        dense_layer = layers.Dense(1, activation="sigmoid")(dropout_layer)
    else:
        dense_layer = layers.Dense(num_classes, activation="softmax")(dropout_layer)

    return Model(inputs=inputs, outputs=dense_layer)


def run_cnn(
    project_name: str,
    kernel_size: Tuple[int, int] = (3, 3),
    pooling_size: int = 2,
    pooling_stride: int = 2,
    cnn_neurons: List[int] = [16, 32, 64, 128],
    dropout_rate: float = 0.3,
    dense_layer_neurons: int = 512,
    projects_path: str = "projects",
) -> Any:
    """
    Run Convolutional Neural Network

    Args:
        project_name: Name of project
        kernel_size: Size of the kernel (Tuple[int, int])
        max_pooling_size: Pooling size of max pooling layers
        max_pooling_stried: Pooling stride of max pooling layers
        cnn_neurons: Number of neurons at each cnn layer
        dropout_rate: Dropout rate
        dense_layer_neurons: Number of neurons at the fully connected dense layer
        projects_path: Projects Path. Defaults to "projects".

    Returns:
        CNN Model
    """
    train_dataset, test_dataset = get_project_datasets(project_name, projects_path)

    model = __create_model(
        project_name,
        kernel_size,
        pooling_size,
        pooling_stride,
        cnn_neurons,
        dropout_rate,
        dense_layer_neurons,
        projects_path,
    )

    model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    return model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=25,
        verbose="binary_accuracy",
    )
