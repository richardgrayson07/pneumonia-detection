# Pneumonia Detection using X-ray Images

This project uses a convolutional neural network (CNN) to classify chest X-ray images into two categories: "Pneumonia" and "Normal".

## Project Overview
- The model is trained using a dataset of chest X-ray images.
- The goal of the project is to predict whether a given X-ray image indicates pneumonia or is normal.

## Requirements
Before running the code, make sure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- Other dependencies listed in `requirements.txt`

## How to Use
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/pneumonia-detection.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure you have the necessary model and images in the respective folders.
4. Run the script:
    ```bash
    python pneumonia_detection.py
    ```

### Model
The model is stored in the `model/` folder. It is a trained CNN model that classifies chest X-ray images. The trained model is available in both `.h5` (HDF5) and `.keras` formats.

### Dataset
You can download the chest X-ray dataset (for training and testing) from [insert link to the dataset]. Ensure that you place the training and testing images in the appropriate folders.

### Example Image
You can test the model with a sample X-ray image placed in the `images/` folder. Make sure to update the image path in the script.

## Author
Sarthak
