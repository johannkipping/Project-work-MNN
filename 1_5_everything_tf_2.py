import warnings

import numpy             as np
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import tensorflow  as tf
import matplotlib.pyplot as plt

from models import FinalModelTF2, FinalModel
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
      'epochs': 15
}

model_param_dict = {
      'activation': 'relu',
      'initializer': 'he_uniform',
      'num_classes': 10,
      'eta': 0.001
}


finmodel = FinalModel(name='final', drop_prob=0.25, **model_param_dict)

model = tfk.models.Sequential()
model.title = 'tf2_final'
model.learning_rate = model_param_dict['eta']
model.add(tfk.layers.InputLayer(input_shape=(32,32,3)))
for layer in finmodel.layers:
    model.add(layer)
    
model.compile(
      optimizer=finmodel.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

info_str = train_and_evaluate(
      model,
      **train_param_dict,
      **data_dict,
      acc_bool=True
)

model.save_weights('./final_model_weights_tf2' )

plt.savefig('./plott.png')
plt.clf()