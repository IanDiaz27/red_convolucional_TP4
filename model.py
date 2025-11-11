from keras import models, preprocessing, layers, utils
import tensorflow as tf
import numpy as np
import pathlib



def crearRed(cant_salidas, input_shape=(200,200,3), seed=11):
    utils.set_random_seed(seed)

    act_conv2D = None
    act_dense = 'relu'
    padding_pooling = 'valid'
    padding_conv2D = 'same'

    # capas de modificación de datos aleatoria
    augment = models.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.15, seed=seed), 
            layers.RandomZoom(0.15, seed=seed), 
            layers.RandomContrast(0.10, seed=seed), 
        ],
        name="data_augmentation"
    )


    model = models.Sequential(
        [
            # Capa de entrada
            layers.Input(shape=input_shape),
            # augment,

            # capas convolutivas
            layers.Conv2D(32, kernel_size=(7,7), strides=(1,1), padding=padding_conv2D, activation=act_conv2D),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2), padding=padding_pooling),

            layers.Conv2D(64, kernel_size=(5,5), strides=(1,1), padding=padding_conv2D, activation=act_conv2D),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2), padding=padding_pooling),

            layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding=padding_conv2D, activation=act_conv2D),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2), padding=padding_pooling),

            layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding=padding_conv2D, activation=act_conv2D),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2), padding=padding_pooling),


            # capas densas
            layers.Flatten(),

            layers.Dense(256, activation=act_dense),
            layers.Dense(128, activation=act_dense),

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



    imprimir_detalle(f"Imagen\t\t\t\tClase\tProb\t\t\t\tOk") 
    cant_ok = 0
    cant_error = 0
    cant_total = 0
    for ruta, idx, prob in zip(paths, indices, probs if 'probs' in locals() else [None]*len(indices)):
        cant_total += 1
        ok = idx == clase_esperada
        if ok:
            cant_ok +=1
        else:
            cant_error +=1
        imprimir_detalle(f"{ruta}\t{idx+1}\t{prob}\t{ok}")

    imprimir_resultado(f"Clase {clase_esperada+1}: Ok: {cant_ok} ({round((cant_ok / cant_total) * 100)}%)\tError: {cant_error} ({round((cant_error / cant_total) * 100)}%) ")


def imprimir_detalle(str, fname="resultados/detalle.txt"):
    print(str, file=open(fname, 'a')) 

def imprimir_resultado(str, fname="resultados/resultados.txt"):
    print(str, file=open(fname, 'a')) 

def obtenerArquitectura(model):
    arquitectura_info = ""
    
    for i, capa in enumerate(model.layers):
        capa_info = f"{i + 1}: {capa.__class__.__name__}"

       
        if isinstance(capa, layers.Conv2D):
            capa_info += f" - Filtros: {capa.filters} Kernel: {capa.kernel_size}, Stride: {capa.strides}, Activ: {capa.activation.__name__ if capa.activation else 'None'}"
        elif isinstance(capa, layers.MaxPooling2D):
            capa_info += f" - Pool Size: {capa.pool_size}, Padding: {capa.padding}"
        elif isinstance(capa, layers.BatchNormalization):
            capa_info += " - Normalización"
        elif isinstance(capa, layers.Dense):
            capa_info += f" - Unidades: {capa.units}, Activ: {capa.activation.__name__ if capa.activation else 'None'}"
        elif isinstance(capa, layers.InputLayer):
            capa_info += f" - Input Shape: {capa.input_shape}"

        arquitectura_info += capa_info + "\n"

    return arquitectura_info.strip()  # Elimina el último salto de línea