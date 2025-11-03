from keras import models, layers, optimizers, metrics, utils, activations
import numpy as np

from model import cargarImagenesEntrenamiento, cargarImagenesValidacion

seed = 11
ds_validacion, clases = cargarImagenesEntrenamiento("img", seed=seed, batch_size=1)
model = models.load_model("punto-1.keras")
resultado = model.predict(ds_validacion)

def equal(a, b):
    if len(a) != len(b):
        return False
    
    for i in range(len(a)):
        if a[i] != b[i]:
            return False

    return True

i = 0
cantDiferentes = 0
for (imgs, labels) in ds_validacion:
    labeObtenido = np.round(resultado[i])
    for label in labels:
        labelEsperado = np.round(label.numpy())
        print(i, labelEsperado , labeObtenido)
        if not equal( labelEsperado, labeObtenido ):
            cantDiferentes += 1
        i += 1


print(i, cantDiferentes, (cantDiferentes/i)*100)