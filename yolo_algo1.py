import tensorflow as tf
import keras
import numpy as np

import os

# Specify the directory where your images are located
directory = r"C:\Users\kusagra\Desktop\deep learning\ssAI\dataset"

# Loop through all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # Check if the file doesn't already have an extension
    if not filename.lower().endswith('.png'):
        # Rename the file by adding the .png extension
        new_filename = f"{filename}.png"
        os.rename(file_path, os.path.join(directory, new_filename))

print("Extensions added successfully!")
