import warnings

import numpy             as np
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import matplotlib.pyplot as plt

from models import InitModel
from custom_utils import train_and_evaluate

# Load and reformat Fashion MNIST dataset 
fashion_mnist = tfk.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images[:,:,:,np.newaxis]
train_images = train_images / 255.0
test_images = test_images[:,:,:,np.newaxis]
test_images = test_images / 255.0

train_labels = tfk.utils.to_categorical(train_labels)
test_labels = tfk.utils.to_categorical(test_labels)

# path were figures will be saved
impath = './img_1_1_init/'
if not os.path.isdir(impath):
    os.makedirs(impath)

# train parameters and data will be constant for all nets
train_param_dict = {
      'batch_size': 128,
      'epochs': 10
}

data_dict = {
      'train_images': train_images,
      'train_labels': train_labels,
      'test_images': test_images,
      'test_labels': test_labels
}

### ZERO INITIALIZATION
# Set up hyperparameters and model parameters
model_param_dict = {
      'activation': 'relu',
      'initializer': 'zeros',
      'num_classes': 10,
      'eta': 0.001
}

# compilation of model
zero_model = InitModel('zero_model', **model_param_dict)
zero_model.compile(
      optimizer=zero_model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

info_str = train_and_evaluate(zero_model, **train_param_dict, **data_dict, info_str='')

### He INITIALIZATION
# Set up hyperparameters and model parameters
model_param_dict['initializer'] = 'he_uniform'

# compilation of model
random_model = InitModel('random_model', **model_param_dict)
random_model.compile(
      optimizer=random_model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

info_str = train_and_evaluate(random_model, **train_param_dict, **data_dict, info_str=info_str)

### KNOWN DATA INITIALIZATION
# Set the all but last two dense layer to be frozen

for layer in random_model.layers[:-2]:
      layer.trainable = False
random_model.title = 'known_data_model'
random_model.compile(
      optimizer=random_model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

info_str = train_and_evaluate(random_model, **train_param_dict, **data_dict, info_str=info_str)

plt.savefig(impath + 'acc_plot_init.png')
plt.clf()