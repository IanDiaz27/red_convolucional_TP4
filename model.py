from keras import models, preprocessing, layers, utils
import tensorflow as tf
import numpy as np
import pathlib



def crearRed(cant_salidas, input_shape=(200,200,3), seed=11):
    utils.set_random_seed(seed)

    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.30, seed=seed), 
            layers.RandomZoom(0.15, seed=seed), 
            layers.RandomContrast(0.5, seed=seed), 
        ],
        name="data_augmentation"
    )


    act_conv2D = None
    act_dense = 'relu'
    padding_pooling = 'same'
    padding_conv2D = 'same'

    model = models.Sequential(
        [
            # Capa de entrada
            layers.Input(shape=input_shape),

            augment,
            
            # Capas de convolución
            layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding=padding_conv2D, activation=act_conv2D),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2), padding=padding_pooling),

            layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding=padding_conv2D, activation=act_conv2D),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2), padding=padding_pooling),

            layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding=padding_conv2D, activation=act_conv2D),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2), padding=padding_pooling),

            layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding=padding_conv2D, activation=act_conv2D),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2), padding=padding_pooling),


            layers.Flatten(),

            layers.Dense(128, activation=act_dense),
            layers.Dense(64, activation=act_dense),

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
        one_hot=True,
        validation_split=0.2):
    
    ds_entrenamiento, ds_valicacion = preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(alto,ancho),
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
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



def cargarImagenesPrediccion(
        path="datos/test",
        alto=200,
        ancho=200,
        batch_size=32,
        normalizar=True):


    path_obj = pathlib.Path(path)
    paths = sorted([str(p) for p in path_obj.rglob("*") if p.is_file()])


    ds_pred = preprocessing.image_dataset_from_directory(
        directory=path,
        labels=None,
        label_mode=None,
        color_mode="rgb",
        image_size=(alto, ancho),
        shuffle=False,
        batch_size=batch_size,
        interpolation="bilinear"
    )

    if normalizar:
        def _norm(imgs):
            return tf.image.convert_image_dtype(imgs, tf.float32)

        ds_pred = ds_pred.map(lambda imgs: _norm(imgs),
                              num_parallel_calls=tf.data.AUTOTUNE)

    # ds_pred = ds_pred.prefetch(tf.data.AUTOTUNE)

    return ds_pred, paths


def predecir(model, imgs, paths, clase_esperada, mostrar_errores=False):
    preds = model.predict(imgs, batch_size=32)

    indices = np.argmax(preds, axis=1)
    probs = np.max(preds, axis=1) 


    if mostrar_errores:
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
        if mostrar_errores:
            print(f"{ruta}\t{idx+1}\t{prob}")

    print(f"**** Clase: {clase_esperada+1}:\t Ok:{cant_ok}\tError: {cant_error}\tPorcentaje Aciertos: {round((cant_ok / cant_total) * 100)} %  *****")


