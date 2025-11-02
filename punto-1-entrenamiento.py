from model import crearRed, cargarImagenes, obtenerMatrizClasificacion

ds_entrenamiento, ds_validacion = cargarImagenes("img")

clases = ds_entrenamiento.class_names
cantClases = len(clases)



print("Clases:", clases)
def count_per_class(dataset):
    counts = {name: 0 for name in clases}
    for images, labels in dataset.unbatch():
        counts[clases[labels.numpy()]] += 1
    return counts

print("Entrenamiento:", count_per_class(ds_entrenamiento))
print("Validación:", count_per_class(ds_validacion))

respuestasEsperadas = obtenerMatrizClasificacion(cantClases)
print(respuestasEsperadas)

# model = crearRed(cantClases)


