from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

app = FastAPI()

# Load the best performing model (assume it's Nested U-Net for this example)
model = load_model("nested_unet_final.h5", custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient})

def preprocess_image(image):
    # Implement preprocessing steps (CLAHE, normalization) here
    # This should match the preprocessing done during training
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Reshape for model input
    input_image = preprocessed_image.reshape(1, 256, 256, 1)  # Adjust shape as needed
    
    # Make prediction
    prediction = model.predict(input_image)
    
    # Convert prediction to binary mask
    binary_mask = (prediction > 0.5).astype(np.uint8).squeeze()
    
    # Encode binary mask as PNG
    _, encoded_mask = cv2.imencode('.png', binary_mask * 255)
    
    return JSONResponse(content={
        "segmentation": encoded_mask.tobytes().decode('latin1')
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)