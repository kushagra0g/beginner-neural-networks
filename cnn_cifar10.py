import tensorflow as tf
import keras
import numpy as np
import math
import matplotlib.pyplot as plt

# defining a show function, to ease showing images, with labels.
def show(x, y, index):
    plt.ion()
    plt.pause(0.001)
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])
    plt.show()

# loading the dataset.
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# reshaping y, since it is pointless for it to be a matrix.
y_train=y_train.reshape(-1)

# defining classes make it easier for us to label the images.
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# converting it to floating points and dividing by 255 to scale it.
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

"""

ANN MODEL

# creating the model and compiling it.
ann_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(3000, activation="relu"),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(10, activation="sigmoid")
        ])

ann_model.compile(optimizer="SGD",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# training the model.
ann_model.fit(x_train, y_train, epochs=2)

# evaluating the model.
ann_model.evaluate(x_train, y_train)


# building and compiling the model.
cnn_model = keras.Sequential([
            # Convolutional layer 1
            keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape = (32,32,3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2,2)),

            # Convolutional layer 2
            keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D((2,2)),

            # Convolutional layer 3
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"),
            keras.layers.BatchNormalization(),

            #Flattening the layers
            keras.layers.Flatten(),

            #Dense layers
            keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation="softmax")
        ])

cnn_model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"]
                  )

# training the model.
cnn_model.fit(x_train, y_train, epochs=20)

# evaluating the model.
cnn_model.evaluate(x_test, y_test)

# saving the model
cnn_model.save(filepath="saved_neural_networks\\cnn_cifar10.keras")"""

cnn_model = keras.models.load_model(filepath="saved_neural_networks\\saved_neural_networks_cnn_cifar10.keras")
cnn_model.evaluate(x_test, y_test)