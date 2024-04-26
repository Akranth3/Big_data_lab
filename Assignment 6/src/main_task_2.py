from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
import PIL
import PIL.Image
import PIL.ImageOps  
import uvicorn
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pickle
pickle_file = "../models/model2.pkl"
with open(pickle_file, "rb") as f:
    model = pickle.load(f)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_image(image: PIL.Image.Image) -> np.ndarray:
    """
    Formats the uploaded image to a 28x28 grayscale image and creates a serialized array of 784 elements.

    Parameters:
    - image (PIL.Image.Image): The uploaded image.

    Returns:
    - np.ndarray: The formatted image as a serialized array of 784 elements.
    """
    # Convert the image to grayscale
    image = image.convert("L")
    
    # Invert the image
    image = 255 - np.array(image)
    
    # Convert the image to a numpy array
    image = np.array(image)
    
    # Pad the image with zeros
    padded_image = np.pad(image, ((1, 1), (8, 8)), mode='constant', constant_values=0)
    
    # Print the shape of the padded image
    print(padded_image.shape)
    
    # Create a new array to store the processed image
    new_array = []
    
    # Iterate over each pixel in the image
    for i in range(28):
        row = []
        for j in range(28):
            # Take the maximum value in each 7x7 block of pixels
            row.append(np.max(padded_image[7*i:7*i+7, 7*j:7*j+7]))
        new_array.append(row)
    
    # Convert the new array to a numpy array
    new_array = np.array(new_array)
    
    # Threshold the image to convert it to black and white
    new_array = np.where(new_array > 40, 255, 0)
    
    # Return the processed image
    return new_array

def load_model(path: str) -> Sequential:
    """
    Load a serialized model from the given path.

    Parameters:
    path (str): The path to the serialized model file.

    Returns:
    Sequential: The loaded model.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_digit(model: Sequential, image: list) -> str:
    """
    Predicts the digit in the given image using the provided model.

    Args:
        model (Sequential): The trained model used for prediction.
        image (list): The image data as a list.

    Returns:
        str: The predicted digit as a string.
    """
    return str(int(np.argmax(model.predict(np.array(image).reshape(1, 784)))))


def predict_digit(model: Sequential, image: list) -> str:
    """
    Predicts the digit in the given image using the provided model.

    Parameters:
    - model (Sequential): The trained model used for prediction.
    - image (list): The image data as a list.

    Returns:
    - str: The predicted digit as a string.
    """
    return str(int(np.argmax(model.predict(np.array(image).reshape(1, 784)))))


def predict_digit(model: Sequential, image: list) -> str:
    """
    Predicts the digit in the given image using the specified model.

    Parameters:
    - model (Sequential): The trained model used for prediction.
    - image (list): The image data as a list.

    Returns:
    - str: The predicted digit as a string.
    """
    return str(int(np.argmax(model.predict(np.array(image).reshape(1, 784)))))



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the digit in an uploaded image file.

    Parameters:
    - file: UploadFile object representing the uploaded image file.

    Returns:
    - A dictionary with the predicted digit.

    Raises:
    - None.
    """
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents))
    image_array = format_image(pil_image)
    model = load_model("../models/model2.pkl")
    digit = predict_digit(model, image_array)
    return {"digit": digit}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)