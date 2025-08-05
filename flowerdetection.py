import PIL.Image
import tensorflow as tf
import keras
import cv2
from PIL import Image
import path
import numpy as np
from keras import Sequential
from sklearn.model_selection import train_test_split
import os

flowers=["daisy","dandelion","roses","sunflowers","tulips"]
data_dir=r"datasets\flower_photos\flower_photos"
data_dir= pathlib.Path(data_dir)
flowers_images = {
                "daisy" : list(data_dir.glob("daisy/*")),
                "dandelion" : list(data_dir.glob("dandelion/*")),
                "roses" : list(data_dir.glob("roses/*")),
                "sunflowers" : list(data_dir.glob("sunflowers/*")),
                "tulips" : list(data_dir.glob("tulips/*"))
                }
flowers_labels = {
                "daisy" : 0,
                "dandelion" : 1,
                "roses" : 2,
                "sunflowers" : 3,
                "tulips" : 4
                }

X, Y = [], []
for flower_name, images in flowers_images.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_image = cv2.resize(img,(180,180))
        X.append(resized_image)
        Y.append(flowers_labels[flower_name])

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255


cnn_file_path = r"C:\Users\kusagra\Desktop\deep learning\codebasicsTutorial\saved_neural_networks\saved_neural_networks_cnn_cifar10.keras"
cnn_file_path = pathlib.Path(cnn_file_path)
flower_model = keras.models.load_model(str(cnn_file_path))
flower_model.summary()

'''elif not cnn_file_path.exists():
    flower_model = Sequential([
        # Convolutional layer 1
        keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(180, 180, 3)),
        keras.layers.MaxPool2D(),

        # Convolutional layer 2
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPool2D(),

        # Convolutional layer 3
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPool2D(),

        # Convolutional layer 4
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPool2D(),

        # Convolutional layer 5
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),

        # Dense Layers
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(5, activation="softmax")
    ])

    flower_model.compile(
        metrics=["accuracy"],
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    )

    flower_model.fit(X_train, Y_train, epochs=30)
    flower_model.evaluate(X_test, Y_test)

    keras.models.save_model(flower_model, filepath=r"./flower_model.keras")'''