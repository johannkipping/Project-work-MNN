 
import tensorflow as tf

print(tf.keras.backend.image_data_format())

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:, :, :, tf.newaxis] / 255.0
x_test = x_test[:, :, :, tf.newaxis] / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                     input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.optimizers import SGD

model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

import numpy as np

p = model.predict(x_test)
guess, label = np.argmax(p, axis=1), np.argmax(y_test, axis=1)
matched = guess == label
CNN_test_accuracy = np.sum(matched) * 100.0 / 10000.0
print('Accuracy on test data:', CNN_test_accuracy)

import matplotlib.pyplot as plt

indices = np.arange(10000).astype(int)
for index in indices[np.logical_not(matched)]:
    plt.imshow(x_test[index, :], 'gray')
    plt.title('guess: {guess}, label: {label}'.\
                  format(guess=guess[index], label=label[index]))
    plt.show()
