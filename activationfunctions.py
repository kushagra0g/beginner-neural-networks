import math
import tensorflow as tf

# dl pt. 8
def sigmoid(r):
    return 1/(1+math.exp(-r))

def tanh(r):
    return (math.exp(r) - math.exp(-r))/(math.exp(r) + math.exp(-r))

def relu(r):
    return max(0,r)

def leaky_relu(r):
    return max(0.1*r,r)


print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU devices:", tf.config.list_physical_devices('GPU'))
#Detailed check
print(tf.sysconfig.get_build_info())