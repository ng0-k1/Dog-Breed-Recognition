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


#Estructura datos de entrada
IMG_SHAPE = (128, 128, 3)
#Uso de red neuronal preentrenada
model_preentrenado = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, 
                                                            include_top=False, 
                                                            weights='imagenet')
model_preentrenado.summary()

model_preentrenado.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(model_preentrenado.output)



prediction_layer = tf.keras.layers.Dense(units=133, activation='softmax')(global_average_layer)
model = tf.keras.models.Model(inputs=model_preentrenado.input, outputs=prediction_layer)

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
              loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

data_gen_train = ImageDataGenerator(rescale = 1/255.)  #, rotation_range = 20, width_shift_range = 0.2, height_shift_range=0.2, zoom_range = 0.1
data_gen_valid = ImageDataGenerator(rescale = 1/255.)
data_gen_test = ImageDataGenerator(rescale = 1/255.)



train_generator = data_gen_train.flow_from_directory(train_dir, target_size = (128, 128), 
                                                     batch_size = 32, class_mode = 'sparse')

validation_generator = data_gen_valid.flow_from_directory(validation_dir, target_size = (128, 128),
                                                          batch_size = 32, class_mode = 'sparse')

test_generator = data_gen_test.flow_from_directory(test_dir, target_size = (128, 128),
                                                          batch_size = 32, class_mode = 'sparse')

#model.fit(train_generator, epochs=8, validation_data=validation_generator)





early_stopping = tf.keras.callbacks.EarlyStopping(patience=4, min_delta=0.01,
                                                  restore_best_weights=True)


history =  model.fit(x=train_generator, epochs=12, 
                     validation_data=validation_generator, batch_size = 16, 
                     callbacks=[early_stopping])

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


#Fine Tuning para reentrenar el modelo despues de determinada capa
fine_tune_at = 120

for layer in model_preentrenado.layers[:fine_tune_at]:
    layer.trainable = False


model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])



history =  model.fit(x=train_generator, epochs=12, 
                     validation_data=validation_generator, batch_size = 16, 
                     callbacks=[early_stopping])
