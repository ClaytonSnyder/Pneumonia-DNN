"""
Vision Transformer Neural Network
"""

from typing import Any

import tensorflow as tf

from einops.layers.tensorflow import Rearrange
from keras import Model, layers
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


def __create_model(
    width: int,
    height: int,
    channels: int,
    patch_size: int,
    num_layers,
    d_model,
    num_heads,
    mlp_dim,
    dropout_rate=0.1,
):
    input_shape = (width, height, channels)

    # Input
    inputs = layers.Input(shape=input_shape)

    # Patch embedding
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    patch_dim = input_shape[-1] // (patch_size**2)
    patches = layers.Conv2D(
        patch_dim, kernel_size=patch_size, strides=patch_size, padding="valid"
    )(inputs)

    # Reshape the outputs of the convolutional layer
    # (batch_size, height, width, channels) -> (batch_size, num_patches, patch_dim)
    patches = Rearrange("b h w c -> b (h w) c")(patches)

    # Positional embeddings
    position_embeddings = layers.Embedding(input_dim=num_patches, output_dim=d_model)(
        tf.range(num_patches)
    )

    # Add positional embeddings to patches
    embeddings = layers.Add()([patches, position_embeddings])

    # Transformer Encoder
    for _ in range(num_layers):
        # Multi-Head Self-Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(embeddings, embeddings)
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(
            attention_output + embeddings
        )

        # Feed Forward Neural Network (MLP)
        mlp_output = layers.Dense(mlp_dim, activation="relu")(attention_output)
        mlp_output = layers.Dense(d_model)(mlp_output)
        mlp_output = layers.Dropout(dropout_rate)(mlp_output)
        mlp_output = layers.LayerNormalization(epsilon=1e-6)(
            mlp_output + attention_output
        )

        embeddings = mlp_output

    # Classification head
    cls_token = embeddings[:, 0]
    outputs = layers.Dense(2, activation="softmax")(cls_token)

    # Model
    model = Model(inputs=inputs, outputs=outputs)
    return model


def run_model(
    projects_name: str,
    projects_path: str,
    width: int,
    height: int,
    channels: int,
    patch_size,
    num_layers: int,
    d_model: int,
    num_heads: int,
    mlp_dim: int,
    dropout_rate: float,
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

    model = __create_model(
        width,
        height,
        channels,
        patch_size,
        num_layers,
        d_model,
        num_heads,
        mlp_dim,
        dropout_rate,
    )
    model.compile(
        optimizer=RMSprop(lr=0.0001),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    return model.fit(train_dataset, validation_data=test_dataset, epochs=25, verbose=1)
