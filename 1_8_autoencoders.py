import warnings

import numpy             as np
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import matplotlib.pyplot as plt
import tensorflow  as tf

from models import Autoencoder
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
      'num_classes': 10,
      'eta': 0.001
}

testim_ind = 0

# Loading model
model = Autoencoder(name='autoencoder_3', latent_dim=3, **model_param_dict)

model.compile(
      optimizer=model.optimizer, 
      loss='mean_squared_error',
      metrics=['accuracy']
)

model.load_weights('./models/autoencoder_3')



# plt.figure()
# plt.subplot(1,3,1)
# plt.xticks([])
# plt.yticks([])
# plt.grid(False)
# plt.imshow(baseImage[0,:,:,:])
# plt.xlabel('Image\nGuess: ' + str(label_init) + '\nConfidence: ' +str(round(confidence_init,2)))
# plt.subplot(1,3,2)
# plt.xticks([])
# plt.yticks([])
# plt.grid(False)
# plt.imshow(delta[0,:,:,:]*100)
# plt.xlabel('Delta*100')
# plt.subplot(1,3,3)
# plt.xticks([])
# plt.yticks([])
# plt.grid(False)
# plt.imshow(adverImage[0,:,:,:])
# plt.xlabel('Adversarial\nGuess: ' + str(label_adversarial) + '\nConfidence: ' +str(round(confidence_adversarial,2)))
# #plt.show()


# plt.savefig(impath + 'adversarial')
# plt.clf()