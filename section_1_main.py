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


# Set up hyperparameters and model parameters
model_param_dict = {
      'activation': 'relu',
      'initializer': 'he_uniform',
      'input_shape': (28,28,1),
      'num_classes': 10,
      'eta': 0.001
}

train_param_dict = {
      'batch_size': 128,
      'epochs': 10
}


# compilation of model
first_model = InitModel('Init_model', **model_param_dict)
first_model.compile(
      optimizer=first_model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

data_dict = {
      'train_images': train_images,
      'train_labels': train_labels,
      'test_images': test_images,
      'test_labels': test_labels
}

train_and_evaluate(first_model, **train_param_dict, **data_dict)