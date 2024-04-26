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
    Predicts the digit in the given image using the provided model.

    Args:
        model (Sequential): The trained model used for prediction.
        image (list): The image data as a list.

    Returns:
        str: The predicted digit as a string.
    """
    return str(int(np.argmax(model.predict(np.array(image).reshape(1, 784)))))

def predict_digit(model: Sequential, image: list) -> str:
    
    return int(np.argmax(model.predict(np.array(image).reshape(1, 784))))


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
    pil_image = pil_image.resize((28, 28))
    image_array = np.array(pil_image).reshape(1, -1)
    
    model = load_model("../models/model2.pkl")
    digit = predict_digit(model, image_array)
    return {"digit": digit}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
