import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt

# loading the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# scaling and converting to float
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

# adding an extra dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# making the model and compiling it.
cnn_mnist = keras.Sequential([
            # Convolutional layer 1
            keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=(28,28, 1)),
            keras.layers.MaxPool2D((2,2)),

            # Convolutional layer 2
            keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
            keras.layers.MaxPool2D((2,2)),

            # Convolutional layer 3
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"),

            # Flattening layer
            keras.layers.Flatten(),

            # Dense layers
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
            ])

print(tf.config.list_physical_devices('GPU'))
cnn_mnist.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer="adam")

# training the model
cnn_mnist.fit(x_train, y_train, epochs=5)

# evaluating the model
cnn_mnist.evaluate(x_test, y_test)