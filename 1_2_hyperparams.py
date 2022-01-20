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
impath = './1_2_hyper/'
if not os.path.isdir(impath):
    os.makedirs(impath)

# data will be same for all nets
data_dict = {
      'train_images': train_images,
      'train_labels': train_labels,
      'test_images': test_images,
      'test_labels': test_labels
}

# train parameters initialization
train_param_dict = {
      'batch_size': 32,
      'epochs': 5
}

# Set up hyperparameters and model parameters
model_param_dict = {
      'activation': 'relu',
      'initializer': 'he_uniform',
      'num_classes': 10,
      'eta': 0.01
}

batch_sizes = [64, 128, 256]
learning_rates = [0.01, 0.001, 0.0001]

for epochs in [5,20]:
      for bs in batch_sizes:
            for eta in learning_rates:
                  train_param_dict['batch_size'] = bs
                  train_param_dict['epochs'] = epochs
                  model_param_dict['eta'] = eta

                  name = 'Model_' + str(bs) + '_' + str(eta) + '_' + str(epochs)
                  
                  # compilation of model
                  model = InitModel(name, **model_param_dict)
                  model.compile(
                        optimizer=model.optimizer, 
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                  )

                  info_str = train_and_evaluate(
                        model,
                        **train_param_dict,
                        **data_dict
                  )
                  plt.savefig(impath + 'acc_plot_' + model.title + '.png')
                  plt.clf()
