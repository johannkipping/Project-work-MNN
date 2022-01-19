import warnings

import numpy             as np
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk

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
      'input_shape': (28,28,1),
      'num_classes': 10,
      'eta': 0.001
}

# compilation of model
zero_model = InitModel('Zero_model', **model_param_dict)
zero_model.compile(
      optimizer=zero_model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

train_and_evaluate(zero_model, **train_param_dict, **data_dict)


### He INITIALIZATION
# Set up hyperparameters and model parameters
model_param_dict['initializer'] = 'he_uniform'

# compilation of model
random_model = InitModel('Random_model', **model_param_dict)
random_model.compile(
      optimizer=random_model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

train_and_evaluate(random_model, **train_param_dict, **data_dict)

### KNOWN DATA INITIALIZATION
# Set the all but last two dense layer to be frozen

for layer in random_model.layers[:-2]:
      layer.trainable = False
random_model.name = 'Known_data_model'
random_model.compile(
      optimizer=random_model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

train_and_evaluate(random_model, **train_param_dict, **data_dict)