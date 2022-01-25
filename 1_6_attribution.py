import warnings

import numpy             as np
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import matplotlib.pyplot as plt
import tensorflow  as tf
from scipy.ndimage import zoom
tf.enable_eager_execution()

from model_builders import get_final_model
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

model_param_dict = {
      'activation': 'relu',
      'initializer': 'he_uniform',
      'num_classes': 10,
      'eta': 0.001
}

testim_ind = 0

# Loading model
model = get_final_model(name='final', **model_param_dict)

model.compile(
      optimizer=model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

model.load_weights('./models/final_fmnist')


# Layer activation visualization
layer_outputs = [layer.output for layer in model.layers[:18]]
activation_model = tfk.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_images[testim_ind][np.newaxis,:,:,:])

plt.figure(figsize=(10,10))
plt.title('Max. mean activations')
for i in range(18):
    layer_activation = activations[i][0,:,:]
    max_ac_ind = np.argmax(np.mean(layer_activation, axis=(0,1)))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(layer_activation[:,:,max_ac_ind], cmap='viridis')
    plt.xlabel('Layer ' + str(i))
plt.show()


# saliency map
test_img = test_images[testim_ind][np.newaxis,:,:,:]
image = tf.Variable(test_img,dtype=float)
with tf.GradientTape() as tape:
    pred = model(image, training=False)
    loss = pred[0][tf.argmax(pred[0])]
grad = tape.gradient(loss, image)

dgrad_max = np.max(tf.math.abs(grad), axis=3)[0]
arr_min, arr_max = np.min(dgrad_max), np.max(dgrad_max)
grad_norm = (dgrad_max-arr_min)/(arr_max-arr_min + 1e-18)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(test_images[testim_ind], cmap='viridis')
plt.xlabel('Test image')
plt.subplot(1,2,2)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(grad_norm[:,:], cmap='viridis')
plt.xlabel('Saliency map')
plt.show()


# Grad-CAM
grad_model = tf.keras.models.Model(
        [model.layers[0].input], [model.layers[14].output, model.layers[-1].output]
)
with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(image)
    pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, pred_index]
grads = tape.gradient(top_class_channel, last_conv_layer_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

heatmap = last_conv_layer_output * pooled_grads
heatmap = tf.reduce_sum(heatmap, axis=(0,3))
heatmap = heatmap.numpy()

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(test_images[testim_ind])
plt.xlabel('Test image')
plt.subplot(1,2,2)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(heatmap, cmap='viridis')
plt.xlabel('Grad-CAM')
plt.show()
