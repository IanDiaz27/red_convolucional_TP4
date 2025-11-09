from keras import metrics, optimizers, callbacks
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf

from model import crearRed, cargarImagenesEntrenamiento


num_prueba = input("Número de Prueba: ")
epocas = 100
batch_size = 32
n = 0.001
seed = 11
loss="sparse_categorical_crossentropy"
# loss="categorical_crossentropy"
# loss="binary_crossentropy"

ds_entrenamiento, ds_validacion, clases = cargarImagenesEntrenamiento(
    path="datos/entrenamiento", 
    seed=seed, 
    batch_size=batch_size)

cantClases = len(clases)
print("Clases:", clases)


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
                validation_data=ds_validacion
            )
fin = time.perf_counter()
# model.save(f"punto-1.keras")

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




