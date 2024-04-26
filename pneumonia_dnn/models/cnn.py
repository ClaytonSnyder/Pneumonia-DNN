"""
Convolutional Neural Network
"""

import time

from typing import Any, List, Tuple

import numpy as np

from keras import Input, Model, layers
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import confusion_matrix

from pneumonia_dnn.utils import (
    get_augmented_inputs,
    get_labels_and_predictions,
    get_project_configuration,
    get_project_datasets,
    save_train_session,
)


def create_model(
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

    return Model(inputs=inputs, outputs=dense_layer), num_classes


def run_cnn(
    project_name: str,
    kernel_size: Tuple[int, int] = (3, 3),
    pooling_size: int = 2,
    pooling_stride: int = 2,
    cnn_neurons: List[int] = [16, 32, 64, 128],
    dropout_rate: float = 0.3,
    dense_layer_neurons: int = 512,
    learning_rate: float = 0.0001,
    epochs: int = 25,
    batch_size: int = 32,
    use_adam_optimizer: bool = True,
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
        learning_rate: Learning rate of loss function
        epochs: Number of epochs
        batch_size: Batch size
        use_adam: If true adam optimizer is used. If false, RMSProp is used
        projects_path: Projects Path. Defaults to "projects".

    Returns:
        CNN Train Session Results
    """
    train_dataset, test_dataset = get_project_datasets(project_name, projects_path)

    model, num_classes = create_model(
        project_name,
        kernel_size,
        pooling_size,
        pooling_stride,
        cnn_neurons,
        dropout_rate,
        dense_layer_neurons,
        projects_path,
    )

    if num_classes > 2:
        loss_function = "categorical_crossentropy"
        metric = "accuracy"
    else:
        loss_function = "binary_crossentropy"
        metric = "binary_accuracy"

    if use_adam_optimizer:
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[metric],
    )

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        verbose=metric,
        batch_size=batch_size,
    )

    true_labels, predictions = get_labels_and_predictions(
        test_dataset, model, batch_size, num_classes
    )

    cm = confusion_matrix(true_labels, predictions)

    loss, accuracy = model.evaluate(test_dataset)

    save_train_session(
        "cnn",
        time.strftime("%Y-%m-%dT%H%M%S"),
        project_name,
        model,
        history.history,
        {
            "kernel_size": kernel_size,
            "pooling_size": pooling_size,
            "pooling_stride": pooling_stride,
            "cnn_neurons": cnn_neurons,
            "dropout_rate": dropout_rate,
            "dense_layer_neurons": dense_layer_neurons,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        projects_path,
    )

    return model, history, predictions, cm, loss, accuracy
