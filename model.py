from keras import models, preprocessing, layers
import numpy as np


def crearRed(cant_clases):
    act="relu"

    model = models.Sequential()
    model.add(layers.Input(shape=(200, 200, 3)))
    model.add(layers.Conv2D(kernel_size=(3,3), padding="same", strides=(1,1), filters=32))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))
    # model.add(layers.Conv2D(kernel_size=(3,3), padding="same", strides=(1,1), filters=64))
    # model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation=act))
    model.add(layers.Dense(128, activation=act))
    model.add(layers.Dense(cant_clases, activation="softmax"))

    return model

def cargarImagenes(path="img"):
    return preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(200,200),
        shuffle=True,
        seed=17,
        validation_split=0.20,            # 20 % para validación
        subset="both",                     # devuelve (train, validation)
    )

def obtenerMatrizClasificacion(cantClases):
    return np.eye(cantClases, dtype=float)


