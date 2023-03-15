# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 22:06:19 2023

@author: Oscar P
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



dataset_path = 'D:/Proyectos_IA/Reconocimiento_Perros/dogImages'
train_dir = os.path.join(dataset_path, 'train')
validation_dir = os.path.join(dataset_path, 'valid')
test_dir = os.path.join(dataset_path, 'test')

## -------------------------------------------- ESTRUCTURA DEL MODELO PREENTRENADO + CAPA DE SALIDA -------- ##
#Estructura datos de entrada
IMG_SHAPE = (128, 128, 3)
#Uso de red neuronal preentrenada
model_preentrenado = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, 
                                                            include_top=False, 
                                                            weights='imagenet')
model_preentrenado.summary()

#Congelando el entrenamiento de la red neuronal preentrenada
model_preentrenado.trainable = False

#Configurando la capa de salida como la media de las salidas
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(model_preentrenado.output)


#Units = 133 porque son 133 razas y una activación por probabilidad de aparación
prediction_layer = tf.keras.layers.Dense(units=256, activation='softmax')(global_average_layer)
model = tf.keras.models.Model(inputs=model_preentrenado.input, outputs=prediction_layer)

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
              loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


## --------------------------------- GENERACIÓN DE IMAGENES Y REESCALADO -------------------------- ##
data_gen_train = ImageDataGenerator(rescale = 1/255.)  #, rotation_range = 20, width_shift_range = 0.2, height_shift_range=0.2, zoom_range = 0.1
data_gen_valid = ImageDataGenerator(rescale = 1/255.)
data_gen_test = ImageDataGenerator(rescale = 1/255.)


#Asignando el directorio y el objetivo que en e ste caso es 128 x 128 que es el tamaño de la imagen
#El class_mode es 'sparse' porque es para datos multiclase

batch_size_data_generator = 16
train_generator = data_gen_train.flow_from_directory(train_dir, target_size = (128, 128), 
                                                     batch_size = batch_size_data_generator, class_mode = 'sparse')

validation_generator = data_gen_valid.flow_from_directory(validation_dir, target_size = (128, 128),
                                                          batch_size = batch_size_data_generator, class_mode = 'sparse')

test_generator = data_gen_test.flow_from_directory(test_dir, target_size = (128, 128),
                                                          batch_size = batch_size_data_generator, class_mode = 'sparse')

#model.fit(train_generator, epochs=8, validation_data=validation_generator)




## -------------------------------------------- PARADA TEMPRANA Y ENTRENAMIENTO DEL MODELO ------------------- ##
early_stopping = tf.keras.callbacks.EarlyStopping(patience=6, min_delta=0.05,
                                                  restore_best_weights=True, mode = 'max')


history =  model.fit(x=train_generator, epochs=20, 
                     validation_data=validation_generator, batch_size = 16, 
                     callbacks=[early_stopping])


##----------------------------------------- EVALUACIÓN DEL MODELO ----------------------------------#
evaluacion = model.evaluate(test_generator)
#predict = model.predict(test_generator)
#predict_t = np.argmax(predict, axis = 1)
#test_names = os.listdir('dogImages/test')
#test = test_generator.next()[1]
#for i in range(len(predict_t)):
#    print(f'Prediccion: {test_names[predict_t[i]]}, \nTest: {test_names[int(test[i])]} \n')

    
#Por revisar
#Problemas existentes respecto a la estimación de la evaluación - - - - Mala Clasificación de datos 
#Revisar el problema del flow_from_directory, solo esta reconociendo unas cuantas clases
#revisar por qué toma el batch_size del flow_from_directory y no del fit directamente



##----------------------------------- APLICANDO FINE TUNING PARA REENTRENAR EL MODELO ----------------------##
#Fine Tuning para reentrenar el modelo despues de determinada capa
fine_tune_at = 90    #120

#Aquí se descongelan son las capas que tiene el modelo, en este caso se entrenaran las ultimas 35 capas de la red neuronal
#TENGA EN CUENTA QUE EL MODELO PREENTRENADO TIENE 155 CAPAS 
for layer in model_preentrenado.layers[:fine_tune_at]:
    layer.trainable = False



## ----------------------------------- COMPILANDO Y REENTRENANDO EL MODELO CON FINE TUNING ---------------- #
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])



history =  model.fit(x=train_generator, epochs=12, 
                     validation_data=validation_generator, batch_size = 16, 
                     callbacks=[early_stopping])

evaluacion = model.evaluate(test_generator)

model.save('modelo_v2_0.h5')


model_v1 = tf.keras.models.load_model('modelo_v2_0.h5')
model_v1.evaluate(test_generator)