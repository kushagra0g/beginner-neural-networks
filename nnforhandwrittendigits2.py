import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

# Converting to float values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Scaling the tuples, for better accuracy and converting them to float
X_train = X_train / 255
X_test = X_test / 255

#Initializing the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Training the model.
model.fit(X_train, Y_train, epochs=25)

#Evaluating the model
model.evaluate(X_test, Y_test)

#Saving the model
model.save(filepath="saved_neural_networks\\final_neural_network_forDigitClassification.h5")

#Prediction and printing it
model_prediction = model.predict(X_train)
image_index=int(input("Enter the image index you want to test. Should be less than 60000\n"))
print("Actual value from dataset = "+str(Y_train[image_index]))
prediction_of_user_given_index=np.argmax(model_prediction[image_index])
print("Predicted value = "+str(prediction_of_user_given_index))