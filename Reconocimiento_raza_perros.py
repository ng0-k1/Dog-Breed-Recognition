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


class DogBreedClassifier:
    """
    Clasificador de razas de perros utilizando MobileNetV2 como base preentrenada.
    Permite entrenamiento inicial y fine-tuning posterior, incluyendo parada temprana y evaluación del modelo.
    """

    def __init__(self, dataset_path: str, img_shape=(128, 128, 3), batch_size=16):
        """
        Inicializa las rutas de los datos, la forma de la imagen y el tamaño de batch.
        
        Args:
            dataset_path (str): Ruta principal al dataset con carpetas 'train', 'valid' y 'test'.
            img_shape (tuple): Dimensiones de las imágenes de entrada.
            batch_size (int): Tamaño del batch usado por los generadores.
        """
        self.dataset_path = dataset_path
        self.train_dir = os.path.join(dataset_path, 'train')
        self.validation_dir = os.path.join(dataset_path, 'valid')
        self.test_dir = os.path.join(dataset_path, 'test')
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.model = None
        self.model_preentrenado = None

        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None

    def prepare_data_generators(self):
        """
        Crea generadores de imágenes para entrenamiento, validación y prueba, aplicando reescalado.
        """
        data_gen = ImageDataGenerator(rescale=1/255.)

        self.train_generator = data_gen.flow_from_directory(
            self.train_dir, target_size=self.img_shape[:2],
            batch_size=self.batch_size, class_mode='sparse')

        self.validation_generator = data_gen.flow_from_directory(
            self.validation_dir, target_size=self.img_shape[:2],
            batch_size=self.batch_size, class_mode='sparse')

        self.test_generator = data_gen.flow_from_directory(
            self.test_dir, target_size=self.img_shape[:2],
            batch_size=self.batch_size, class_mode='sparse')

    def build_model(self, num_classes=133):
        """
        Construye el modelo con MobileNetV2 como base, usando una capa de promedio global y una capa de salida.
        
        Args:
            num_classes (int): Número de clases de salida (razas de perros).
        """
        self.model_preentrenado = tf.keras.applications.MobileNetV2(
            input_shape=self.img_shape, include_top=False, weights='imagenet')
        self.model_preentrenado.trainable = False

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(self.model_preentrenado.output)
        prediction_layer = tf.keras.layers.Dense(units=256, activation='softmax')(global_average_layer)

        self.model = tf.keras.models.Model(inputs=self.model_preentrenado.input, outputs=prediction_layer)

        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )

    def train_model(self, epochs=20, patience=6):
        """
        Entrena el modelo utilizando parada temprana basada en la precisión de validación.
        
        Args:
            epochs (int): Número máximo de épocas.
            patience (int): Número de épocas sin mejora para activar parada temprana.
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=patience, min_delta=0.05, restore_best_weights=True, mode='max')

        return self.model.fit(
            x=self.train_generator,
            validation_data=self.validation_generator,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping]
        )

    def evaluate_model(self):
        """
        Evalúa el modelo sobre el conjunto de prueba.
        
        Returns:
            tuple: Pérdida y precisión del modelo en el conjunto de prueba.
        """
        return self.model.evaluate(self.test_generator)

    def fine_tune_model(self, fine_tune_at=90, learning_rate=0.001, epochs=12):
        """
        Descongela parte del modelo preentrenado y realiza fine-tuning.
        
        Args:
            fine_tune_at (int): Índice de la capa desde la cual descongelar.
            learning_rate (float): Tasa de aprendizaje durante el fine-tuning.
            epochs (int): Número de épocas de reentrenamiento.
        """
        for layer in self.model_preentrenado.layers[:fine_tune_at]:
            layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=6, min_delta=0.05, restore_best_weights=True, mode='max')

        return self.model.fit(
            x=self.train_generator,
            validation_data=self.validation_generator,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping]
        )

    def save_model(self, path='modelo_v2_0.h5'):
        """
        Guarda el modelo entrenado en el disco.
        
        Args:
            path (str): Ruta donde guardar el archivo del modelo.
        """
        self.model.save(path)

    def load_model(self, path='modelo_v2_0.h5'):
        """
        Carga un modelo previamente guardado.
        
        Args:
            path (str): Ruta del archivo del modelo a cargar.
        """
        self.model = tf.keras.models.load_model(path)


# ----------------------- USO EJEMPLO --------------------------- #
if __name__ == "__main__":
    classifier = DogBreedClassifier(dataset_path='D:/Proyectos_IA/Reconocimiento_Perros/dogImages')
    classifier.prepare_data_generators()
    classifier.build_model()
    classifier.train_model()
    classifier.evaluate_model()

    # Fine-tuning opcional
    classifier.fine_tune_model()
    classifier.evaluate_model()

    classifier.save_model()
