import warnings

import numpy             as np
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import matplotlib.pyplot as plt
import tensorflow  as tf

from models import Autoencoder, AutoencoderConv
from custom_utils import plot_latent_space_interpolation, plot_mnist_latent_space


# Load and reformat Fashion MNIST dataset 
fashion_mnist = tfk.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images[:,:,:,np.newaxis]
train_images = train_images / 255.0
test_images = test_images[:,:,:,np.newaxis]
test_images = test_images / 255.0

#train_labels = tfk.utils.to_categorical(train_labels)
#test_labels = tfk.utils.to_categorical(test_labels)

# path were figures will be saved
impath = './img_1_8_autoencoder/'
if not os.path.isdir(impath):
    os.makedirs(impath)

# data will be same for all nets
data_dict = {
      'train_images': train_images,
      'train_labels': train_images,
      'test_images': test_images,
      'test_labels': test_images
}

model_param_dict = {
      'activation': 'relu',
      'initializer': 'he_uniform',
      'eta': 0.001
}

# Loading model
model = Autoencoder(name='autoencoder_2', latent_dim=2, **model_param_dict)
model.compile(
      optimizer=model.optimizer, 
      loss='mean_squared_error',
      metrics=['accuracy']
)

model.load_weights('./models/autoencoder_2')

encoded_imgs_3 = model.encoder(test_images).numpy()
decoded_imgs_3 = model.decoder(encoded_imgs_3).numpy()

plot_mnist_latent_space(test_images, test_labels, class_names, model.encoder, impath + 'autoencoder_latent_2', 2)
plot_latent_space_interpolation(
    model.encoder,
    test_images,
    test_labels,
    model.decoder,
    impath + 'autoencoder_interpolate',
    scale_x=30.0,
    scale_y=30.0,
)

# Loading model
model = Autoencoder(name='autoencoder_64', latent_dim=64, **model_param_dict)
model.compile(
      optimizer=model.optimizer, 
      loss='mean_squared_error',
      metrics=['accuracy']
)

model.load_weights('./models/autoencoder_64')

encoded_imgs_64 = model.encoder(test_images).numpy()
decoded_imgs_64 = model.decoder(encoded_imgs_64).numpy()


# Loading model
model = AutoencoderConv(name='autoencoder_conv', **model_param_dict)
model.compile(
      optimizer=model.optimizer, 
      loss='mean_squared_error',
      metrics=['accuracy']
)

model.load_weights('./models/autoencoder_conv')

encoded_imgs_conv = model.encoder(test_images).numpy()
decoded_imgs_conv = model.decoder(encoded_imgs_conv).numpy()

n = 10
plt.figure(figsize=(20,10),tight_layout=True)
plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.95, wspace=0.01, hspace=0.35)
for i in range(n):
  # display original
  ax = plt.subplot(4, n, i + 1)
  plt.imshow(test_images[i])
  plt.title("original")
  plt.gray()
  plt.xticks([])
  plt.yticks([])

  # display reconstruction 3
  ax = plt.subplot(4, n, i + 1 + n)
  plt.imshow(decoded_imgs_3[i])
  plt.title("Latent dim. 3")
  plt.gray()
  plt.xticks([])
  plt.yticks([])
  
  # display reconstruction 64
  ax = plt.subplot(4, n, i + 1 + 2*n)
  plt.imshow(decoded_imgs_64[i])
  plt.title("Latent dim. 64")
  plt.gray()
  plt.xticks([])
  plt.yticks([])
  
  # display reconstruction convolution
  ax = plt.subplot(4, n, i + 1 + 3*n)
  plt.imshow(decoded_imgs_conv[i])
  plt.title("Convolution")
  plt.gray()
  plt.xticks([])
  plt.yticks([])
#plt.show()

plt.savefig(impath + 'autoencoder_test')
plt.clf()
