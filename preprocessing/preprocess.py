import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    images = []
    masks = []
    logger.info(f"Searching for images and masks in: {data_path}")
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.tif') and not file.endswith('_mask.tif'):
                image_path = os.path.join(root, file)
                mask_path = os.path.join(root, file.replace('.tif', '_mask.tif'))
                
                logger.info(f"Found image: {image_path}")
                if os.path.exists(mask_path):
                    logger.info(f"Found corresponding mask: {mask_path}")
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    if image is not None and mask is not None:
                        images.append(image)
                        masks.append(mask)
                        logger.info("Successfully loaded image and mask.")
                    else:
                        logger.warning(f"Failed to load image or mask: {image_path}")
                else:
                    logger.warning(f"No corresponding mask found for: {image_path}")
    
    logger.info(f"Total images loaded: {len(images)}")
    logger.info(f"Total masks loaded: {len(masks)}")
    return np.array(images), np.array(masks)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def augment_data(image, mask):
    # Implement data augmentation techniques
    # For simplicity, we'll just add horizontal flip for now
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask

def preprocess_dataset(data_path):
    images, masks = load_data(data_path)
    
    if len(images) == 0 or len(masks) == 0:
        raise ValueError("No images or masks found in the specified directory.")
    
    preprocessed_images = []
    preprocessed_masks = []
    
    for image, mask in zip(images, masks):
        # Apply CLAHE
        enhanced_image = apply_clahe(image)
        
        # Normalize
        normalized_image = normalize_image(enhanced_image)
        
        # Augment
        augmented_image, augmented_mask = augment_data(normalized_image, mask)
        
        preprocessed_images.append(augmented_image)
        preprocessed_masks.append(augmented_mask)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_images, preprocessed_masks, test_size=0.2, random_state=42
    )
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data", "raw")
    logger.info(f"Using data path: {data_path}")
    
    try:
        X_train, X_test, y_train, y_test = preprocess_dataset(data_path)
        
        # Create processed directory if it doesn't exist
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        
        # Save preprocessed data
        np.save("data/processed/X_train.npy", X_train)
        np.save("data/processed/X_test.npy", X_test)
        np.save("data/processed/y_train.npy", y_train)
        np.save("data/processed/y_test.npy", y_test)
        
        logger.info(f"Preprocessed data saved. Shapes:")
        logger.info(f"X_train: {X_train.shape}")
        logger.info(f"X_test: {X_test.shape}")
        logger.info(f"y_train: {y_train.shape}")
        logger.info(f"y_test: {y_test.shape}")
    except ValueError as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.info("Please ensure that your data directory contains .tif images and their corresponding _mask.tif files.")