## 1 - PACKAGES ##

# Import personnal Functions
from myfunctions import *

# Import Keras functions
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD


## 2 - Exo 2 ##

# Builidng of the logistic regression
L = 100
model = Sequential()
model.add(Dense(L,  input_dim=784, name='fc1'))
model.add(Dense(10, name='fc2'))
model.add(Activation('softmax'))

# Display the number of parameters of the model
print(model.summary())

# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the images as vectors
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Convert values as float an rescale
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Number of classes
K = 10

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)

# optimiser of the model
learning_rate = 10e-3
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])

# Fitting of the data
batch_size = 100
nb_epoch = 50
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Display the scores
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Save the model
saveModel(model, "Results/Model")

# Display some weights
