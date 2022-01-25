import warnings

import numpy             as np
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import matplotlib.pyplot as plt
import tensorflow  as tf

from models import FinalModelTF2
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
finmodel = FinalModelTF2(name='final', drop_prob=0.25, **model_param_dict)

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

model.load_weights('./final_model_weights_tf2')


def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return  tf.reduce_sum(losses)

class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[32,32,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8 

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients*step_size
            img = tf.clip_by_value(img, 0, 1)

        return loss, img
    
def run_deep_dream_simple(img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    #img = tfk.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))

    return img


# Maximize the activations of these layers
names = ['conv2d']
names = ['conv2d_2', 'conv2d_5']
layers = [model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=model.input, outputs=layers)


deepdream = DeepDream(dream_model)

test_img = tf.cast(test_images[13], tf.float32)

dream_img = test_img
base_shape = tf.shape(dream_img)[:-1]
plt.figure()
for i in range(100):
    dream_img = run_deep_dream_simple(img=dream_img, steps=100, step_size=0.001*i/100)
    #figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(dream_img)
    plt.savefig(impath + 'movie/' + str(i))
    plt.clf()   
    #dream_img = tf.image.resize(dream_img, base_shape).numpy()
for i in range(100,300):
    dream_img = run_deep_dream_simple(img=dream_img, steps=100, step_size=0.002)
    #plt.figure()#figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(dream_img)
    plt.savefig(impath + 'movie/' + str(i))
    plt.clf()   
    dream_img = dream_img[1:-1, 1:-1, :]
    dream_img = tf.image.resize(dream_img, base_shape).numpy()

# #######################################################
# # OCTAVE SCALING
# import time
# start = time.time()

# OCTAVE_SCALE = 1.30

# img = tf.constant(np.array(test_img))
# base_shape = tf.shape(img)[:-1]
# float_base_shape = tf.cast(base_shape, tf.float32)

# for n in range(-2, 3):
#   new_shape = tf.cast(float_base_shape*(OCTAVE_SCALE**n), tf.int32)

#   img = tf.image.resize(img, new_shape).numpy()
#   img = tf.image.resize(img, base_shape).numpy()

#   img = run_deep_dream_simple(img=img, steps=50, step_size=0.001)

# img = tf.image.resize(img, base_shape)

# end = time.time()
# end-start
# #####################################################


plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(test_img)
plt.xlabel('Test image')
plt.subplot(1,2,2)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(dream_img)
plt.xlabel('Deep Dream')
#plt.show()


plt.savefig(impath + names[0])
plt.clf()