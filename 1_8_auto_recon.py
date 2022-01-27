import warnings
import copy

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

# what will be left at the edges
cutsize = 7
train_images_cut = copy.copy(train_images) 
train_images_cut[:, cutsize:28-cutsize, cutsize:28-cutsize, :] = 0
test_images_cut = copy.copy(test_images) 
test_images_cut[:, cutsize:28-cutsize, cutsize:28-cutsize, :] = 0

#train_labels = tfk.utils.to_categorical(train_labels)
#test_labels = tfk.utils.to_categorical(test_labels)

# path were figures will be saved
impath = './img_1_8_autoencoder/'
if not os.path.isdir(impath):
    os.makedirs(impath)

# data will be same for all nets
data_dict = {
      'train_images': train_images_cut,
      'train_labels': train_images,
      'test_images': test_images_cut,
      'test_labels': test_images
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

# Loading model
model = AutoencoderConv(name='autoencoder_cut', **model_param_dict)
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
model.save_weights('./models/autoencoder_cut')

encoded_imgs_cut = model.encoder(test_images_cut).numpy()
decoded_imgs_cut = model.decoder(encoded_imgs_cut).numpy()

n = 10
plt.figure(figsize=(20,10),tight_layout=True)
plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.95, wspace=0.01, hspace=0.35)
for i in range(n):
  # display original
  ax = plt.subplot(3, n, i + 1)
  plt.imshow(test_images[i])
  plt.title("Original")
  plt.gray()
  plt.xticks([])
  plt.yticks([])

  # display reconstruction 3
  ax = plt.subplot(3, n, i + 1 + n)
  plt.imshow(test_images_cut[i])
  plt.title("Cut original")
  plt.gray()
  plt.xticks([])
  plt.yticks([])
  
  # display reconstruction 3
  ax = plt.subplot(3, n, i + 1 + 2*n)
  plt.imshow(decoded_imgs_cut[i])
  plt.title("Reconstruction")
  plt.gray()
  plt.xticks([])
  plt.yticks([])
  
#plt.show()

plt.savefig(impath + 'auto_reconstruction')
plt.clf()