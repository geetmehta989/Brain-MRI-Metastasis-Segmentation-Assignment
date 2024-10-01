import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import io

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose a brain MRI image...", type="png")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Make prediction
    files = {'file': ('image.png', img_byte_arr, 'image/png')}
    response = requests.post("http://localhost:8000/predict", files=files)
    
    if response.status_code == 200:
        # Decode the segmentation mask
        segmentation = np.frombuffer(response.json()['segmentation'].encode('latin1'), dtype=np.uint8)
        segmentation = cv2.imdecode(segmentation, cv2.IMREAD_GRAYSCALE)

        # Create a color overlay
        color_mask = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        color_mask[segmentation > 0] = [255, 0, 0]  # Red color for segmentation

        # Blend original image with color mask
        original_image = np.array(image.convert('RGB'))
        blended = cv2.addWeighted(original_image, 0.7, color_mask, 0.3, 0)

        # Display the result
        st.image(blended, caption='Segmentation Result', use_column_width=True)
    else:
        st.error("Error in segmentation. Please try again.")

st.write("Upload a brain MRI image to see the metastasis segmentation result.")