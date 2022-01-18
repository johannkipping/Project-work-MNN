import numpy             as np
import matplotlib.pyplot as plt
import tensorflow.keras  as tfk

from random import randrange
from time   import time      # For time measuring


fashion_mnist = tfk.datasets.fashion_mnist
# x_train, y_train = fashion_mnist['x_train'].reshape(60000,28,28), fashion_mnist['y_train']
# x_test,  y_test  = fashion_mnist['x_test'].reshape(10000,28,28), fashion_mnist['y_test']
# x_train, x_test  = x_train / 255.0, x_test / 255.0

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


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

model = tfk.Sequential()

model.add(tfk.Input(shape=input_shape))
model.add(tfk.layers.Conv2D(32, (3,3),
                             activation=activation,
                             kernel_initializer='he_uniform'))
model.add(tfk.layers.MaxPool2D(pool_size=(2,2)))
model.add(tfk.layers.Conv2D(64, (3,3),
                             activation=activation,
                             kernel_initializer='he_uniform'))
model.add(tfk.layers.MaxPool2D(pool_size=(2,2)))
model.add(tfk.layers.Flatten())
model.add(tfk.layers.Dense(2048, activation=activation,
                            kernel_initializer='he_uniform'))
model.add(tfk.layers.Dense(10, activation='softmax',
                            kernel_initializer='he_uniform'))

opt = tfk.optimizers.Adam(eta)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=opt, 
      #loss='categorical_crossentropy',
      loss=tfk.losses.SparseCategoricalCrossentropy(),
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

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
