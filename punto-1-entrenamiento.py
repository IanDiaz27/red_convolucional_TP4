from keras import metrics, optimizers, callbacks
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf

from model import crearRed, cargarImagenesEntrenamiento, cargarImagenesValidacion



epocas = 500
batch_size = 8
n = 0.001
seed = 11
ds_entrenamiento, clases = cargarImagenesEntrenamiento("img", seed=seed, batch_size=batch_size)
ds_validacion, _ = cargarImagenesValidacion("img", seed=seed, batch_size=batch_size)

cantClases = len(clases)
print("Clases:", clases)

# inicio de entrenamiento

# tf.config.set_visible_devices([], device_type='GPU')
# print("devices:", tf.config.get_visible_devices())
# tf.config.experimental.enable_op_determinism()




model = crearRed(cantClases, seed)
opt = optimizers.Adam(learning_rate = n)
# metrica = metrics.BinaryAccuracy(name="binary accuracy", dtype=None, threshold=0.5)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

def lr_schedule(epoch, lr):
    if epoch != 0 and epoch % 10 == 0:
        return lr * 0.5
    return lr

lr_scheduler = callbacks.LearningRateScheduler(lr_schedule, verbose=1)

# Opcional: EarlyStopping para evitar sobre‑entrenamiento
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)



inicio = time.perf_counter()
historia = model.fit(ds_entrenamiento, 
                epochs = epocas,
                callbacks=[lr_scheduler, early_stop],
                validation_data=ds_validacion
            )
fin = time.perf_counter()
model.save(f"punto-1.keras")

print(f"tiempo consumido: {fin - inicio}")

plt.plot(historia.history['loss'])
plt.title(f"Entrenamiento - Epocas: {epocas} - Batch Size: {batch_size}")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.ylim(0, 2)
plt.legend(["entrenamiento"],loc="upper right")
plt.savefig(f"punto-1-entrenamiento_loss_{epocas}_{batch_size}_{n}.png")



resultado = model.predict(ds_entrenamiento)

def equal(a, b):
    if len(a) != len(b):
        return False
    
    for i in range(len(a)):
        if a[i] != b[i]:
            return False

    return True

i = 0
cantDiferentes = 0
for (imgs, labels) in ds_entrenamiento:
    for label in labels:
        labeObtenido = np.round(resultado[i])
        labelEsperado = np.round(label.numpy())
        print(i, labelEsperado , labeObtenido)
        if not equal( labelEsperado, labeObtenido ):
            cantDiferentes += 1
        i += 1


print(i, cantDiferentes, (cantDiferentes/i)*100)

# ds_entrenamiento, clases = cargarImagenesValidacion("img", seed=seed)


# model.predict(ds_entrenamiento)


