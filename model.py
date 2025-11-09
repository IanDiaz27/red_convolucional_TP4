from keras import models, preprocessing, layers, utils
import tensorflow as tf
import numpy as np
from pathlib import Path



def crearRed(cant_salidas, input_shape=(200,200,3), seed=11):
    utils.set_random_seed(seed)

    model = models.Sequential(
        [
            # Capa de entrada
            layers.Input(shape=input_shape),
            
            # Capas de convolución
            layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding="same"),
            layers.MaxPooling2D(pool_size=(2,2), padding="same"),

            layers.Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same"),
            layers.MaxPooling2D(pool_size=(2,2), padding="same"),

            layers.Conv2D(128, kernel_size=(7,7), strides=(1,1), padding="same"),
            layers.MaxPooling2D(pool_size=(2,2), padding="same"),

            layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding="same"),
            layers.MaxPooling2D(pool_size=(2,2), padding="same"),

            layers.Flatten(),

            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),

            # 
            # layers.Dropout(0.5),

            # Capa de Salida
            layers.Dense(cant_salidas, activation="softmax"),
        ]
    )

    return model

def cargarImagenesEntrenamiento(
        path="datos/entrenamiento", 
        alto=200, ancho=200, 
        seed=11, batch_size=32,
        normalizar=True,
        one_hot=True ):
    
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

    # normalización entre 0 y 1
    if normalizar:
        def normalizar(imgs, labels):
            imgs = tf.image.convert_image_dtype(imgs, tf.float32) 
            return imgs, labels

        ds_entrenamiento = ds_entrenamiento.map(normalizar, num_parallel_calls=tf.data.AUTOTUNE)
        ds_valicacion = ds_valicacion.map(normalizar, num_parallel_calls=tf.data.AUTOTUNE)

    if one_hot:
        ds_entrenamiento = ds_entrenamiento.map(lambda imgs, labels: (imgs, tf.one_hot(labels, depth=cantClases)),
                                                num_parallel_calls=tf.data.AUTOTUNE)
        ds_valicacion = ds_valicacion.map(lambda imgs, labels: (imgs, tf.one_hot(labels, depth=cantClases)),
                                          num_parallel_calls=tf.data.AUTOTUNE)


    ds_entrenamiento = ds_entrenamiento.prefetch(tf.data.AUTOTUNE)
    ds_valicacion = ds_valicacion.prefetch(tf.data.AUTOTUNE)

    return ds_entrenamiento, ds_valicacion, clases



# carga una imagen 
def cargarImagen(path, alto=200, ancho=200, normalizar=True):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [alto, ancho])
    if normalizar:
        img = tf.image.convert_image_dtype(img, tf.float32)


    return img


def cargarImagenesClasificacion(
    path,
    alto= 200,
    ancho= 200,
    normalizar= True,
    ):
    base = Path(path)

    paths = [p for p in base.iterdir() if p.is_file()]

    imgs = []
    for p in paths:
        img = cargarImagen(str(p), alto, ancho, normalizar)
        imgs.append(img)

    imgs_tensor = tf.stack(imgs, axis=0)

    return imgs_tensor, paths



def predecir(model, imgs, paths, clase_esperada):
    # predecir (devuelve array de predicciones para cada imagen)
    preds = model.predict(imgs, batch_size=32)

    # interpretar salidas (ejemplos)
    if preds.ndim == 2 and preds.shape[1] > 1:
        indices = np.argmax(preds, axis=1)
        probs = np.max(preds, axis=1) 


    print(f"Imagen\t\t\tClase\tProb") 
    cant_ok = 0
    cant_error = 0
    cant_total = 0
    for ruta, idx, prob in zip(paths, indices, probs if 'probs' in locals() else [None]*len(indices)):
        cant_total += 1
        if idx == clase_esperada:
            cant_ok +=1
        else:
            cant_error +=1
        print(f"{ruta}\t{idx+1}\t{prob}")

    print(f"****  Ok:{cant_ok} - Error: {cant_error} - Porcentaje Aciertos: {round((cant_ok / cant_total) * 100, 2)} %  *****")


def cargarImagenesPrediccion(
        path="datos/test",
        alto=200, ancho=200,
        batch_size=32,
        normalizar=True,
        devolver_rutas=True,
    ):
    # label_mode="none" si el directorio sólo contiene imágenes sin subcarpetas de clase
    ds_pred, info = preprocessing.image_dataset_from_directory(
        path,
        labels= None,
        label_mode= None,
        color_mode="rgb",
        image_size=(alto, ancho),
        shuffle=False,
        seed=0,
        batch_size=batch_size
    ), None

    if normalizar:
        def _normalizar(imgs, labels=None):
            imgs = tf.image.convert_image_dtype(imgs, tf.float32)
            return (imgs, labels) if labels is not None else imgs
        ds_pred = ds_pred.map(lambda imgs: _normalizar(imgs), num_parallel_calls=tf.data.AUTOTUNE)

    # Obtener rutas de archivos si se solicita (útil para mapear predicciones)
    path_obj = pathlib.Path(path)
    # Si hay subcarpetas de clase, tomamos todas las imágenes recursivamente
    patrones = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp"]
    archivos = []
    for p in patrones:
        archivos += sorted([str(pth) for pth in path_obj.glob(p)])
    paths = archivos

    ds_pred = ds_pred.prefetch(tf.data.AUTOTUNE)
    return ds_pred, paths