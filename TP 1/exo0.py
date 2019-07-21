## 1 - PACKAGES ##

# Import personnal functions
from functions import *

# Keras Functions
from keras.datasets import mnist

# Import matplotlib
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt


## 2 - Exo 0 ##

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the images as vectors
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Convert values as float an rescale
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Shapes of the data
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Display figures
plt.figure(figsize=(7.195, 3.841), dpi=100)
for i in range(200):
    plt.subplot(10, 20, i+1)
    plt.imshow(X_train[i, :].reshape([28, 28]), cmap='gray')
    plt.axis('off')
plt.show()
