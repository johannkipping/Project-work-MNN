import warnings

import numpy             as np
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk

from models import WideModel
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
      'epochs': 10
}
              
# compilation of model
model = WideModel()
model.compile(
      optimizer=model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

train_and_evaluate(
      model,
      folder_name='1_3_structure',
      **train_param_dict,
      **data_dict
)
