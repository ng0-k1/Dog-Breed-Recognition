# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 19:34:12 2023

@author: Oscar P
"""
import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model


class DogBreedPredictorAPI:
    """
    API REST para la predicción de razas de perros a partir de imágenes.
    Utiliza un modelo previamente entrenado y guardado en formato HDF5.
    """

    def __init__(self, model_path: str, class_dir: str, image_size=(128, 128)):
        """
        Inicializa el predictor cargando el modelo y las etiquetas de clases.

        Args:
            model_path (str): Ruta al archivo del modelo (.h5).
            class_dir (str): Ruta al directorio donde están las clases (subcarpetas).
            image_size (tuple): Tamaño de entrada requerido por el modelo.
        """
        self.model = load_model(model_path, compile=False)
        self.class_names = os.listdir(class_dir)
        self.image_size = image_size

    def preprocess_image(self, file_bytes: bytes) -> np.ndarray:
        """
        Procesa la imagen cargada para que sea compatible con el modelo.

        Args:
            file_bytes (bytes): Bytes de la imagen subida.

        Returns:
            np.ndarray: Imagen preprocesada como array normalizado y redimensionado.
        """
        image = Image.open(io.BytesIO(file_bytes))
        image = image.resize(self.image_size)
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Agrega dimensión batch
        image_array = image_array.astype('float32') / 255.0
        return image_array

    def predict_label(self, image_array: np.ndarray) -> str:
        """
        Realiza la predicción con el modelo y devuelve la etiqueta.

        Args:
            image_array (np.ndarray): Imagen ya preprocesada.

        Returns:
            str: Nombre de la clase predicha.
        """
        prediction = self.model.predict(image_array)
        predicted_index = np.argmax(prediction[0])
        return self.class_names[predicted_index]


# ---------------------- INICIALIZACIÓN DE LA API ---------------------- #
predictor = DogBreedPredictorAPI(
    model_path='modelo_v2_0.h5',
    class_dir='dogImages/test'
)

app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint para predecir la raza de un perro a partir de una imagen.

    Args:
        file (UploadFile): Imagen cargada por el usuario.

    Returns:
        dict: Etiqueta predicha para la imagen.
    """
    contents = await file.read()
    image_array = predictor.preprocess_image(contents)
    label = predictor.predict_label(image_array)
    return {"label": label}
