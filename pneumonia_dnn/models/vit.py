"""
Vision Transformer Implementation
"""

import time

from typing import Any

import keras
import numpy as np
import tensorflow as tf

from keras import layers
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import confusion_matrix

from pneumonia_dnn.utils import (
    get_augmented_inputs,
    get_labels_and_predictions,
    get_project_configuration,
    get_project_datasets,
    save_train_session,
)


def create_multilayer_perceptron(x, hidden_units, dropout_rate):
    """
    Creates multi-layer perceptron

    Args:
        x: Input layer
        hidden_units: Hidden Units
        dropout_rate: Dropout Rate

    Returns:
        Multi-Layer perceptron
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    """
    Patches Layer
    """

    def __init__(self, patch_size, batch_size, height, width, channels):
        super().__init__()
        self.patch_size = patch_size
        self.height = height
        self.width = width
        self.channels = channels
        self.batch_size = batch_size

    def call(self, images):
        images_batch_size = tf.shape(images)[0]  # type: ignore
        num_patches_h = self.height // self.patch_size
        num_patches_w = self.width // self.patch_size
        patches = tf.image.extract_patches(
            images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(
            patches,
            (
                images_batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * self.channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    """
    Patch Encoder LAyer
    """

    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.expand_dims(tf.range(self.num_patches), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)  # type: ignore
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


def create_model(
    project_name: str,
    batch_size: int,
    patch_size: int,
    projection_dim: int,
    num_heads: int,
    transformer_layers: int,
    projects_path: str,
):
    project_data = get_project_configuration(project_name, projects_path)

    height = project_data["image_height"]
    width = project_data["image_width"]
    channels = project_data["image_channels"]
    augmentation = get_augmented_inputs(project_name, projects_path)
    num_classes = len(project_data["labels"])

    num_patches = (width // patch_size) ** 2

    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]

    mlp_head_units = [
        2048,
        1024,
    ]

    inputs = keras.Input(shape=(height, width, channels))
    augmented = augmentation(inputs)

    # Create patches.
    patches = Patches(patch_size, batch_size, height, width, channels)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = create_multilayer_perceptron(
            x3, hidden_units=transformer_units, dropout_rate=0.1
        )
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = create_multilayer_perceptron(
        representation, hidden_units=mlp_head_units, dropout_rate=0.5
    )
    # Classify outputs.
    if num_classes == 2:
        logits = layers.Dense(1, activation="sigmoid")(features)
    else:
        logits = layers.Dense(num_classes, activation="softmax")(features)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model, num_classes


def run_vit(
    project_name: str,
    patch_size: int = 6,
    projection_dim: int = 32,
    num_heads: int = 2,
    transformer_layers: int = 3,
    learning_rate: float = 0.0001,
    epochs: int = 25,
    batch_size: int = 32,
    use_adam_optimizer: bool = True,
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
        use_adam: If true adam optimizer is used. If false, RMSProp is used
        projects_path: _description_. Defaults to "projects".

    Returns:
        VIT Train Session Results
    """
    keras.mixed_precision.set_global_policy("mixed_float16")
    train_dataset, test_dataset = get_project_datasets(project_name, projects_path)

    model, num_classes = create_model(
        project_name,
        batch_size,
        patch_size,
        projection_dim,
        num_heads,
        transformer_layers,
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
        batch_size=batch_size,
    )

    true_labels, predictions = get_labels_and_predictions(
        test_dataset, model, batch_size, num_classes
    )

    cm = confusion_matrix(true_labels, predictions)

    loss, accuracy = model.evaluate(test_dataset)

    save_train_session(
        "vit",
        time.strftime("%Y-%m-%dT%H%M%S"),
        project_name,
        model,
        history.history,
        {
            "patch_size": patch_size,
            "projection_dim": projection_dim,
            "num_heads": num_heads,
            "transformer_layers": transformer_layers,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        projects_path,
    )

    return model, history, predictions, cm, loss, accuracy
