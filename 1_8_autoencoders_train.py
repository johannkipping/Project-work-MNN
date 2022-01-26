import warnings

import numpy             as np
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import matplotlib.pyplot as plt
import tensorflow  as tf

from models import Autoencoder, AutoencoderConv
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
impath = './img_1_8_autoencoder/'
if not os.path.isdir(impath):
    os.makedirs(impath)

# data will be same for all nets
data_dict = {
      'train_images': train_images,
      'train_labels': train_labels,
      'test_images': test_images,
      'test_labels': test_labels
}

model_param_dict = {
      'activation': 'relu',
      'initializer': 'he_uniform',
      'eta': 0.001
}

# train parameters initialization
train_param_dict = {
      'batch_size': 128,
      'epochs': 20
}

testim_ind = 0

# Loading model
model = Autoencoder(name='autoencoder_3', latent_dim=3, **model_param_dict)
model.compile(
      optimizer=model.optimizer, 
      loss='mean_squared_error',
      metrics=['accuracy']
)
info_str = train_and_evaluate(
      model,
      **train_param_dict,
      **data_dict,
      acc_bool=True
)
model.save_weights('./models/autoencoder_3')

plt.savefig(impath + 'auto_training_3')
plt.clf()


# Loading model
model = Autoencoder(name='autoencoder_64', latent_dim=64, **model_param_dict)
model.compile(
      optimizer=model.optimizer, 
      loss='mean_squared_error',
      metrics=['accuracy']
)
info_str = train_and_evaluate(
      model,
      **train_param_dict,
      **data_dict,
      acc_bool=True
)
model.save_weights('./models/autoencoder_64')

plt.savefig(impath + 'auto_training_64')
plt.clf()


# Loading model
model = AutoencoderConv(name='autoencoder_conv', **model_param_dict)
model.compile(
      optimizer=model.optimizer, 
      loss='mean_squared_error',
      metrics=['accuracy']
)
info_str = train_and_evaluate(
      model,
      **train_param_dict,
      **data_dict,
      acc_bool=True
)
model.save_weights('./models/autoencoder_conv')

plt.savefig(impath + 'auto_training_conv')
plt.clf()
