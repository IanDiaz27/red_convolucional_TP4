from keras import models, preprocessing, layers, utils
import tensorflow as tf
import numpy as np


def crearRed(cant_clases, seed=11):
    utils.set_random_seed(seed)

    act="relu"

    model = models.Sequential()
    model.add(layers.Input(shape=(200, 200, 3)))

    model.add(layers.Conv2D(kernel_size=(3,3), padding="same", strides=(1,1), filters=32))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(kernel_size=(3,3), padding="same", strides=(1,1), filters=64))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(kernel_size=(3,3), padding="same", strides=(1,1), filters=128))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(kernel_size=(3,3), padding="same", strides=(1,1), filters=256))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(kernel_size=(3,3), padding="same", strides=(1,1), filters=512))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(kernel_size=(3,3), padding="same", strides=(1,1), filters=64))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))


    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation=act))
    model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(128, activation=act))
    # model.add(layers.Dense(64, activation=act))
    model.add(layers.Dense(cant_clases, activation="softmax"))

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


