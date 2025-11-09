from keras import models, preprocessing, layers, utils
import tensorflow as tf
import numpy as np


def crearRed(cant_salidas, input_shape=(200,200,3), seed=11):
    utils.set_random_seed(seed)

    model = models.Sequential(
        [
            # Capa de entrada
            layers.Input(shape=input_shape),
            
            # Capas de convolución
            layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="same"),
            layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),

            # layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same"),
            # layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),

            layers.Flatten(),

            layers.Dense(200, activation="relu"),
            # layers.Dense(128, activation="relu"),

            # 
            # layers.Dropout(0.5),

            # Capa de Salida
            layers.Dense(cant_salidas, activation="softmax"),
        ]
    )



    return model

def cargarImagenesEntrenamiento(path="img", alto=200, ancho=200, seed=11, batch_size=32):
    ds_entrenamiento, ds_valicacion = preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(alto,ancho),
        shuffle=True,
        seed=seed,
        validation_split=0.20,
        subset="both",
        batch_size=batch_size
    )

    clases = ds_entrenamiento.class_names
    cantClases = len(clases)

    # crea un map agregando una matriz con las respuestas esperadas
    # ds_entrenamiento = ds_entrenamiento.map(lambda imgs, labels: (imgs, tf.one_hot(labels, depth=cantClases)) )
    # ds_valicacion = ds_valicacion.map(lambda imgs, labels: (imgs, tf.one_hot(labels, depth=cantClases)) )

    return ds_entrenamiento, ds_valicacion, clases


