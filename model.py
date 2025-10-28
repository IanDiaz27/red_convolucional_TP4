from keras import models, metrics, layers

act="relu"

model = models.Sequential()
model.add(layers.Input(shape=(200, 200, 3)))
model.add(layers.Conv2D(kernel_size=(3,3), padding="same", strides=(1,1), filters=32))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same"))
model.add(layers.Conv2D(kernel_size=(3,3), padding="same", strides=(1,1), filters=64))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same"))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation=act))
model.add(layers.Dense(128, activation=act))
model.add(layers.Dense(8, activation="softmax"))
