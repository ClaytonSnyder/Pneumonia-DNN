"""
Vision Transformer Neural Network
"""

import tensorflow as tf

from keras import layers, models
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


from einops.layers.tensorflow import Rearrange


def __create_model(
    width: int,
    height: int,
    patch_size: int,
    num_layers,
    d_model,
    num_heads,
    mlp_dim,
    dropout_rate=0.1,
):
    input_shape = (width, height)

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


# Parameters
input_shape = (224, 224, 3)  # Example shape for chest X-rays # Pneumonia or non-pneumonia
patch_size = 16  # Size of patches
num_layers = 6  # Number of transformer layers
d_model = 256  # Model dimension
num_heads = 8  # Number of attention heads
mlp_dim = 512  # MLP hidden dimension
dropout_rate = 0.1  # Dropout rate

# Create the ViT model
model = vision_transformer(
    input_shape,
    patch_size,
    num_layers,
    d_model,
    num_heads,
    mlp_dim,
    dropout_rate,
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Print model summary
model.summary()
