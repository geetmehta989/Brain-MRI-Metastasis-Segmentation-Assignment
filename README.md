# Brain MRI Metastasis Segmentation

This project implements and compares Nested U-Net (U-Net++) and Attention U-Net architectures for brain MRI metastasis segmentation. It includes a web application to showcase the model's performance.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Architectures](#model-architectures)
5. [Results](#results)
6. [Challenges and Solutions](#challenges-and-solutions)
7. [Future Work](#future-work)

## Project Overview

This project aims to segment brain metastases in MRI images using advanced deep learning techniques. We implement and compare two state-of-the-art architectures: Nested U-Net (U-Net++) and Attention U-Net. The project includes data preprocessing, model training, evaluation, and a web application for easy use.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/brain-mri-segmentation.git
   cd brain-mri-segmentation
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Preprocess the data:
   ```
   python preprocessing/preprocess.py
   ```

2. Train the models:
   ```
   python training/train.py
   ```

3. Evaluate the models:
   ```
   python evaluation/evaluate.py
   ```

4. Run the FAST API backend:
   ```
   uvicorn webapp.backend.main:app --reload
   ```

5. Run the Streamlit frontend:
   ```
   streamlit run webapp/frontend/app.py
   ```

6. Open your web browser and navigate to `http://localhost:8501` to use the web application.

## Model Architectures

### Nested U-Net (U-Net++)

Nested U-Net, also known as U-Net++, is an advanced version of the original U-Net architecture. It introduces dense skip connections and deep supervision to improve segmentation accuracy. The nested structure allows the model to capture fine-grained details at various scales, which is particularly useful for detecting small metastases.

### Attention U-Net

Attention U-Net incorporates attention gates into the standard U-Net architecture. These gates help the model focus on relevant features and suppress irrelevant ones. This is especially beneficial for brain metastasis segmentation, as it allows the model to concentrate on small, often subtle metastatic regions while reducing false positives in healthy tissue.

## Results

(Include your model comparison results here, such as DICE scores and example segmentations)

## Challenges and Solutions

1. **Small and Diverse Metastases**: Brain metastases can vary greatly in size and appearance. We addressed this by using data augmentation techniques and implementing architectures (Nested U-Net and Attention U-Net) that can capture multi-scale features.

2. **Limited Dataset**: Medical imaging datasets are often limited in size. We mitigated this by using transfer learning, data augmentation, and employing architectures that perform well with limited data.

3. **Class Imbalance**: Metastases typically occupy a small portion of the brain. We addressed this by using the DICE loss function, which is less affected by class imbalance compared to binary cross-entropy.

4. **Preprocessing Challenges**: MRI images can vary in intensity and contrast. We implemented CLAHE (Contrast Limited Adaptive Histogram Equalization) to standardize image contrast and improve metastasis visibility.

## Future Work

1. Incorporate multi-modal MRI data (T1, T2, FLAIR) to improve segmentation accuracy.
2. Explore other advanced architectures such as TransUNet or Swin-Unet for potentially better performance.
3. Implement uncertainty quantification to provide confidence measures for the segmentations.
4. Collaborate with radiologists to refine the model based on clinical feedback and integrate it into clinical workflows.

## Video Demonstration

(Include a link to your video demonstration here)

---

For any questions or issues, please open an issue in the GitHub repository.