import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
X_train = (X_train.astype("float32"))/255
X_test = (X_test.astype("float32"))/255
model = keras.models.load_model("saved_neural_networks/final_neural_network_forDigitClassification.h5")
prediction=model.predict(X_train)
while True:
    userInput = input("Enter the index number, between 0 and 60000. To exit/stop, input exit. (lowercase)\n")
    if userInput=="exit" or userInput=="no" or userInput=="quit" or userInput=="stop":
        break
    userInput=int(userInput)
    actual_value=Y_train[userInput]
    predicted_value=np.argmax(prediction[userInput])
    print("Predicted value = "+str(predicted_value))
    print("Actual value from dataset = "+str(actual_value))
