import numpy             as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
#import tensorflow.compat.v1  as tf
#from tensorflow import keras

from random import randrange
from time   import time      # For time measuring


fashion_mnist = tfk.datasets.fashion_mnist
# x_train, y_train = fashion_mnist['x_train'].reshape(60000,28,28), fashion_mnist['y_train']
# x_test,  y_test  = fashion_mnist['x_test'].reshape(10000,28,28), fashion_mnist['y_test']
# x_train, x_test  = x_train / 255.0, x_test / 255.0

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images[:,:,:,np.newaxis]
train_images = train_images / 255.0
test_images = test_images[:,:,:,np.newaxis]
test_images = test_images / 255.0

train_labels = tfk.utils.to_categorical(train_labels)
test_labels = tfk.utils.to_categorical(test_labels)


"""
Categories for Fashion MNIST. Category i is ct[i].
"""
ct = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
      'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

"""
Set up hyperparameters
"""
bs, ep, eta = 128, 10, .001
activation = 'relu'


"""
Set up the network with TensorFlow
"""
input_shape=(28,28,1)

inputs = tfk.Input(shape=input_shape)

x = tfk.layers.Conv2D(
    32,
    (3,3),
    activation=activation,
    kernel_initializer='he_uniform'
)(inputs)

x = tfk.layers.MaxPool2D(pool_size=(2,2))(x)

x = tfk.layers.Conv2D(
    64,
    (3,3),
    activation=activation,
    kernel_initializer='he_uniform'
)(x)

x = tfk.layers.MaxPool2D(pool_size=(2,2))(x)

x = tfk.layers.Flatten()(x)

x = tfk.layers.Dense(
    2048,
    activation=activation,
    kernel_initializer='he_uniform'
)(x)

predictions = tfk.layers.Dense(
    10,
    activation='softmax',
    kernel_initializer='he_uniform'
)(x)

model = tfk.Model(inputs=inputs, outputs=predictions)

opt = tfk.optimizers.Adam(eta)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=opt, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

model.summary()

start = time()
history = model.fit(train_images, train_labels, batch_size=bs, epochs=ep, validation_data=(test_images, test_labels))
train_time = time() - start


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('Hyperparameters: batch size:', bs, 'epochs: ', ep, 'eta: ', eta)
print('Accuracy =', test_acc)
print('Time needed for training:', train_time)

print(history.history.keys())

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
