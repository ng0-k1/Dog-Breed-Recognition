# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:12:21 2023

@author: Oscar P
"""

import os
import tempfile
import requests
import json
import tensorflow as tf
import shutil
import subprocess
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


model_recharge = tf.keras.models.load_model('modelo_v2_0.h5')

dataset_path = 'D:/Proyectos_IA/Reconocimiento_Perros/dogImages'
test_dir = os.path.join(dataset_path, 'test')
data_gen_test = ImageDataGenerator(rescale = 1/255.)
X_test = data_gen_test.flow_from_directory(test_dir, target_size = (128, 128),
                                                          batch_size = 32, class_mode = 'sparse')


# Definir el directorio del modelo y la versión
MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join('D:\\Proyectos_IA\\Reconocimiento_Perros\\', 'rec_perro', str(version))

# Verificar si la carpeta del modelo ya existe y eliminarla en caso contrario
if os.path.isdir(export_path):
    shutil.rmtree(export_path)

# Guardar el modelo utilizando la función tf.keras.models.save_model
os.environ['MODEL_DIR'] = os.path.abspath(MODEL_DIR)
tf.keras.models.save_model(
    model=model_recharge,
    filepath=export_path,
    overwrite=True,
    include_optimizer=True
)


# Definir el comando a ejecutar
#command = 'start cmd /c "tensorflow_model_server --rest_api_port=9999 --model_name=rec_perro --model_base_path=%MODEL_DIR% > server.log 2>&1"'
command = 'start cmd /c "tensorflow_model_server --rest_api_port=8888 --model_name=rec_perro --model_base_path={} > server.log 2>&1"'.format(export_path)
# Iniciar el servidor del modelo en segundo plano utilizando la función subprocess.Popen
server = subprocess.Popen(command, shell=True)



#Traer una imagen aleatoria 
random_image = np.random.randint(0, X_test.n)

image_position = random_image
X_test.reset() # reiniciar el generador para comenzar desde el principio
for i in range(image_position + 1):
    images, labels = X_test.next()
image = images[0] # la primera imagen del lote de tamaño 32

# Realizar una solicitud de predicción utilizando la API REST del servidor
data = json.dumps({"signature_name":"serving_default", "instances":[image.tolist()]})



headers = {"content-type":"application/json"}
json_response = requests.post(url='http://localhost:8888/v1/models/rec_perro:predict', data=data, headers=headers)












