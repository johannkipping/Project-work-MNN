import warnings
import sys
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy             as np

import tensorflow.keras  as tfk
import tensorflow  as tf
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_builders import get_final_model
from custom_utils import train_and_evaluate

# Load and reformat Fashion MNIST dataset 
cifar10 = tfk.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#train_images = train_images[:,:,:,np.newaxis]
train_images = train_images / 255.0
#test_images = test_images[:,:,:,np.newaxis]
test_images = test_images / 255.0

train_labels = tfk.utils.to_categorical(train_labels)
test_labels = tfk.utils.to_categorical(test_labels)

# data will be same for all nets
data_dict = {
      'train_images': train_images,
      'train_labels': train_labels,
      'test_images': test_images,
      'test_labels': test_labels
}

# train parameters initialization
train_param_dict = {
      'batch_size': 128,
      'epochs': 30
}

model_param_dict = {
      'activation': 'relu',
      'initializer': 'he_uniform',
      'num_classes': 10,
      'eta': 0.001
}


model = get_final_model(name='final', input_shape=(32,32,3), **model_param_dict)
    
model.compile(
      optimizer=model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

info_str = train_and_evaluate(
      model,
      **train_param_dict,
      **data_dict,
      acc_bool=True
)

model.save_weights('./models/final_cifar10' )

plt.savefig('./models/final_cifar10.png')
plt.clf()