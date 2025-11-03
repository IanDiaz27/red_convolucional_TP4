from keras import models, preprocessing, layers, utils
import tensorflow as tf
import numpy as np


def crearRed(ancho_image, alto_imagen, cant_clases, seed=11):
    utils.set_random_seed(seed)

    model = keras.Sequential(
        [
            layers.Input(shape=(ancho_image, alto_imagen, 3)),
            # data_augmentation,
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(cant_clases, activation="softmax"),
        ]
    )

    return model

def cargarImagenesEntrenamiento(path="img", seed=11, batch_size=32):
    ds = preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(200,200),
        shuffle=True,
        seed=seed,
        validation_split=0.20,
        subset="training",
        batch_size=batch_size
    )

    clases = ds.class_names
    cantClases = len(clases)

    # crea un map agregando una matriz con las respuestas esperadas
    ds = ds.map(lambda imgs, labels: (imgs, tf.one_hot(labels, depth=cantClases)) )

    return ds, clases

def cargarImagenesValidacion(path="img", seed=11, batch_size=32):
    ds = preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(200,200),
        shuffle=True,
        seed=seed,
        validation_split=0.20,
        subset="validation",  
        batch_size=batch_size
    )

    clases = ds.class_names
    cantClases = len(clases)

    # crea un map agregando una matriz con las respuestas esperadas
    ds = ds.map(lambda imgs, labels: (imgs, tf.one_hot(labels, depth=cantClases)) )

    return ds, clases


def obtenerMatrizClasificacion(cantClases):
    return np.eye(cantClases, dtype=float)


