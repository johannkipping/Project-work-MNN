import warnings

import numpy             as np
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import matplotlib.pyplot as plt

from models import FinalModel
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

#train_labels = tfk.utils.to_categorical(train_labels)
#test_labels = tfk.utils.to_categorical(test_labels)

# path were figures will be saved
impath = './img_1_6_attribution/'
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
      'batch_size': 128,
      'epochs': 20
}

model_param_dict = {
      'activation': 'relu',
      'initializer': 'he_uniform',
      'num_classes': 10,
      'eta': 0.001
}


model = FinalModel(name='final', drop_prob=0.25, **model_param_dict)
model.compile(
      optimizer=model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

model.load_weights('./final_model_weights_fashion' )

activations = model.predict(test_images)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()



plt.savefig(impath + 'dropout.png')
plt.clf()
