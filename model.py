act="relu"

model = Sequential()
model.add(Conv2D(input_shape=(200,200,3), kernel_size(3,3), padding="same", strides=(1,1) filters=32))
model.add(MaxPooling2D(pool_size=(2,2), strides(1,1), padding="same"))
model.add(Conv2D(kernel_size(3,3), padding="same", strides=(1,1) filters=64))
model.add(MaxPooling2D(pool_size=(2,2), strides(1,1), padding="same"))
model.add(Flatten())
model.add(Dense(256, activation=act))
model.add(Dense(128, activation=act))
model.add(Dense(8, activation="softmax"))
