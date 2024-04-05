"""
Convolutional Neural Network
"""

from typing import Any

import tensorflow as tf

from keras import layers, models
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


def __create_model(width: int, height: int, channels: int):
    """
    Create Convolutional Neural Network

    Args:
        width: Width of the images
        height: Height of the images
        channels: Number of channels
    """
    return models.Sequential(
        [
            layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=(width, height, channels)
            ),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


def run_model(
    projects_name: str, projects_path: str, width: int, height: int, channels: int
) -> Any:
    train_data_path = f"{projects_path}/{projects_name}/dataset/train"
    test_data_path = f"{projects_path}/{projects_name}/dataset/test"

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_dataset = train_datagen.flow_from_directory(
        train_data_path, target_size=(width, height), batch_size=32, class_mode="binary"
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_dataset = test_datagen.flow_from_directory(
        test_data_path, batch_size=10, class_mode="binary", target_size=(width, height)
    )

    model = __create_model(width, height, channels)
    model.compile(
        optimizer=RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"]
    )

    return model.fit(
        train_dataset, validation_data=test_dataset, epochs=25, verbose=binary_accuracy
    )
