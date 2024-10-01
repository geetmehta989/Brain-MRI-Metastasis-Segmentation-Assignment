import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocessing.preprocess import preprocess_dataset

def dice_coefficient(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def evaluate_model(model_path, X_test, y_test):
    # Load model
    model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient})

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate DICE score
    dice_scores = [dice_coefficient(y_true, y_pred).numpy() for y_true, y_pred in zip(y_test, y_pred)]
    
    return np.mean(dice_scores)

if __name__ == "__main__":
    data_path = "path/to/your/data"
    X_train, X_test, y_train, y_test = preprocess_dataset(data_path)

    nested_unet_path = "nested_unet_final.h5"
    attention_unet_path = "attention_unet_final.h5"

    nested_unet_dice = evaluate_model(nested_unet_path, X_test, y_test)
    attention_unet_dice = evaluate_model(attention_unet_path, X_test, y_test)

    print(f"Nested U-Net DICE Score: {nested_unet_dice:.4f}")
    print(f"Attention U-Net DICE Score: {attention_unet_dice:.4f}")

    # You can add code here to generate and save visualizations of the segmentation results