from keras import metrics, optimizers, callbacks
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

from model import crearRed, cargarImagenesEntrenamiento, cargarImagenesClasificacion, predecir

num_prueba = input("Número de Prueba: ")
epocas = 100
batch_size = 32
n = 0.001
seed = 11

loss="categorical_crossentropy"
# loss="binary_crossentropy"

ds_entrenamiento, ds_validacion, clases = cargarImagenesEntrenamiento(
    path="datos/entrenamiento", 
    seed=seed, 
    batch_size=batch_size)

cantClases = len(clases)
print("Clases:", clases)


# ajustar n
def scheduler(epoch, lr):
    if epoch > 0 and epoch % 10 == 0:
        return lr * 0.5
    return lr

early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=30, 
    restore_best_weights=True,
    verbose=1
)


lr_callback = LearningRateScheduler(scheduler, verbose=1)

# inicio de entrenamiento

# tf.config.set_visible_devices([], device_type='GPU')
# print("devices:", tf.config.get_visible_devices())
tf.config.experimental.enable_op_determinism()

model = crearRed(cant_salidas=cantClases, seed=seed)
opt = optimizers.Adam(learning_rate = n)
model.compile(loss = loss, optimizer = opt, metrics = ['accuracy'])

inicio = time.perf_counter()
historia = model.fit(ds_entrenamiento, 
                epochs = epocas,
                validation_data=ds_validacion,
                callbacks=[lr_callback, early_stop],
        )
fin = time.perf_counter()
model.save(f"{num_prueba}_{epocas}_{batch_size}_{n}_{loss}.keras")

print(f"tiempo consumido: {fin - inicio}")

plt.title(f"Loss - Epocas: {epocas} - Batch Size: {batch_size} - n: {n}")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.plot(historia.history['loss'])
plt.plot(historia.history['val_loss'])
plt.legend(['loss', 'val_loss'],loc="upper right")
plt.savefig(f"resultados/prueba-{num_prueba}_{epocas}_{batch_size}_{n}_{loss}_loss.png")


plt.clf()
plt.title(f"Accuracy - Epocas: {epocas} - Batch Size: {batch_size} - n: {n}")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.plot(historia.history['accuracy'])
plt.plot(historia.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'],loc="upper right")
plt.savefig(f"resultados/prueba-{num_prueba}_{epocas}_{batch_size}_{n}_{loss}_accuracy.png")


# probar clase 1
for clase in range(cantClases):
    imgs, paths = cargarImagenesClasificacion(f"datos/test/{clase+1}", alto=200, ancho=200, normalizar=True)
    predecir(model, imgs, paths, clase)

