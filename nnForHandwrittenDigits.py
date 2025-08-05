import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import numpy as np
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
X_train = X_train.reshape(len(X_train), 28 * 28).astype('float32') / 255.0
X_test = X_test.reshape(len(X_test), 28 * 28).astype('float32') / 255.0
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation="sigmoid")
])
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
    )
model.fit(X_train, y_train, epochs=5)
model.evaluate(X_test, y_test)

# hidden layer
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation="sigmoid")
])
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
    )
model.fit(X_train, y_train, epochs=5)



image_index = 591
x_test_image_at_image_index = X_test[image_index]
y_predicted = model.predict(X_test)
print(np.argmax(y_predicted[image_index]))