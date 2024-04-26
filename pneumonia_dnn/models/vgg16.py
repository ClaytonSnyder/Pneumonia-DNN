"""
Convolutional Neural Network
"""

import time

from typing import Any

from keras import Input, Model, layers
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import confusion_matrix

from pneumonia_dnn.utils import (
    get_augmented_inputs,
    get_labels_and_predictions,
    get_project_configuration,
    get_project_datasets,
    save_train_session,
)


def __create_model(
    project_name: str,
    weights: str,
    include_top: bool,
    dense_layer_neurons: int,
    projects_path: str,
):
    """
    Create Convolutional Neural Network

    Args:
        project_name: Name of project
        weights: one of None (random initialization),
            "imagenet" (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        include_top: Whether to include the 3 fully-connected layers at the top of the network.
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

    inputs = Input(shape=(height, width, channels))
    augmented = augmentation(inputs)

    pre_trained_model = VGG16(
        weights=weights, include_top=include_top, input_shape=(height, width, channels)
    )

    for layer in pre_trained_model.layers:
        layer.trainable = False

    pretrained_outputs = pre_trained_model(augmented)

    flatten_layer = layers.Flatten()(pretrained_outputs)
    dense_layer = layers.Dense(dense_layer_neurons, activation="relu")(flatten_layer)

    if num_classes == 2:
        dense_layer = layers.Dense(1, activation="sigmoid")(dense_layer)
    else:
        dense_layer = layers.Dense(num_classes, activation="softmax")(dense_layer)

    return Model(inputs=inputs, outputs=dense_layer), num_classes


def run_vgg16(
    project_name: str,
    weights: str = "imagenet",
    include_top: bool = False,
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
        weights: one of None (random initialization),
            "imagenet" (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        include_top: Whether to include the 3 fully-connected layers at the top of the network.
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

    model, num_classes = __create_model(
        project_name,
        weights,
        include_top,
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
        "vgg16",
        time.strftime("%Y-%m-%dT%H%M%S"),
        project_name,
        model,
        history.history,
        {
            "weights": weights,
            "include_top": include_top,
            "dense_layer_neurons": dense_layer_neurons,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        projects_path,
    )

    return model, history, predictions, cm, loss, accuracy
