from keras import optimizers, callbacks
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf

from model import crearRed, cargarImagenesEntrenamiento, predecir, cargarImagenesPrediccion

# tf.config.experimental.enable_op_determinism()


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
    batch_size=batch_size,
    normalizar=True,
    validation_split=0.15)

cantClases = len(clases)
print("Clases:", clases)


# ajustar n
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',      # métrica a observar
    factor=0.5,              # LR ← LR * factor
    patience=5,              # epochs sin mejora antes de reducir
    min_lr=1e-6,             # LR mínimo permitido
    verbose=1)

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=30, 
    restore_best_weights=True,
    verbose=1
)



model = crearRed(cant_salidas=cantClases, seed=seed)
opt = optimizers.Adam(learning_rate = n)
model.compile(loss = loss, optimizer = opt, metrics = ['accuracy'])

inicio = time.perf_counter()
historia = model.fit(ds_entrenamiento, 
                epochs = epocas,
                validation_data=ds_validacion,
                callbacks=[reduce_lr],
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
    imgs, paths = cargarImagenesPrediccion(f"datos/test/{clase+1}", alto=200, ancho=200, normalizar=True)
    predecir(model, imgs, paths, clase, False)

