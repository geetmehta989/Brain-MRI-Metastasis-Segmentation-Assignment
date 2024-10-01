import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nested_unet import build_nested_unet
from models.attention_unet import build_attention_unet
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def dice_coefficient(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.keras.backend.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def train_model(model_type, input_shape, X_train, y_train, X_val, y_val, batch_size=8, epochs=100):
    # Build model
    if model_type == "nested_unet":
        model = build_nested_unet(input_shape)
    elif model_type == "attention_unet":
        model = build_attention_unet(input_shape)
    else:
        raise ValueError("Invalid model type. Choose 'nested_unet' or 'attention_unet'.")

    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=[dice_coefficient])

    # Callbacks
    callbacks = [
        ModelCheckpoint(f"best_{model_type}.h5", save_best_only=True, monitor='val_dice_coefficient', mode='max'),
        ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.1, patience=5, min_lr=1e-7, mode='max'),
        EarlyStopping(monitor='val_dice_coefficient', patience=15, mode='max', restore_best_weights=True)
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )

    return model, history

if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Print data types and shapes
    print("X_train dtype:", X_train.dtype, "shape:", X_train.shape)
    print("y_train dtype:", y_train.dtype, "shape:", y_train.shape)

    # Convert data to float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Normalize input data if not already done
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape input data if necessary
    # Assuming the images are grayscale and the shape is (num_images, height, width)
    # We need to add a channel dimension
    if len(X_train.shape) == 3:
        X_train = X_train.reshape((*X_train.shape, 1))
        X_test = X_test.reshape((*X_test.shape, 1))

    # Ensure y_train and y_test are in the correct shape
    if len(y_train.shape) == 3:
        y_train = y_train.reshape((*y_train.shape, 1))
        y_test = y_test.reshape((*y_test.shape, 1))

    input_shape = X_train.shape[1:]  # (height, width, 1)

    print("Final shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # Train Nested U-Net
    nested_unet, nested_history = train_model("nested_unet", input_shape, X_train, y_train, X_test, y_test)

    # Train Attention U-Net
    attention_unet, attention_history = train_model("attention_unet", input_shape, X_train, y_train, X_test, y_test)

    # Save models
    nested_unet.save("nested_unet_final.h5")
    attention_unet.save("attention_unet_final.h5")

    print("Training completed. Models saved.")