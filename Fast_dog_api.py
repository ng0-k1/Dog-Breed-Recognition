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
import tensorflow.keras.models as models



model = models.load_model('D:/Proyectos_IA/Reconocimiento_Perros/modelo_v2_0.h5', compile = False)


app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    #Etiquetas de las imagenes
    class_names = os.listdir('D:/Proyectos_IA/Reconocimiento_Perros/dogImages/test')
    
    # Leer el contenido del archivo
    contents = await file.read()
    # Convertir el contenido del archivo en una imagen PIL
    image = Image.open(io.BytesIO(contents))
    # Convertir la imagen a 128x128 píxeles
    image = image.resize((128, 128))
    # Convertir la imagen a un array numpy
    image_array = np.array(image)
    # Añadir una dimensión adicional para que tenga la forma (1, 128, 128, 3)
    image_array = np.expand_dims(image_array, axis=0)
    # Normalizar los valores de píxel a un rango de 0 a 1
    image_array = image_array.astype('float32') / 255.0
    # Realizar la predicción con el modelo cargado
    prediction = model.predict(image_array)
    # Convertir la predicción en una etiqueta (o lo que sea que el modelo haya sido entrenado para predecir)
    
    label = class_names[np.argmax(prediction[0])]
    #return prediction
    return {"label": label}



