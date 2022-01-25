import warnings

import numpy             as np
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import matplotlib.pyplot as plt
import tensorflow  as tf

from model_builders import get_final_model
from custom_utils import train_and_evaluate


# Load and reformat Fashion MNIST dataset 
cifar10 = tfk.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
model = get_final_model(name='final', **model_param_dict)

model.compile(
      optimizer=model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

model.load_weights('./models/final_cifar10')

#test_loss, test_acc = finmodel.evaluate(test_images,  test_labels, verbose=2)
#prediction_init = model(tf.expand_dims(test_images[0], axis=0))
#print(str(test_loss))
#print(str(test_acc))

def clip_eps(tensor, eps):
    # clip the values of the tensor to a given range and return it
	return tf.clip_by_value(tensor, clip_value_min=-eps, clip_value_max=eps)

def generate_adversaries(model, baseImage, delta, classIdx, steps=50):
    # iterate over the number of steps
	for step in range(0, steps):
		# record our gradients
		with tf.GradientTape() as tape:
			# explicitly indicate that our perturbation vector should
			# be tracked for gradient updates
			tape.watch(delta)
            # add our perturbation vector to the base image and
			# preprocess the resulting image
			adversary = baseImage + delta
			# run this newly constructed image tensor through our
			# model and calculate the loss with respect to the
			# *original* class index
			predictions = model(adversary, training=False)
			loss = -tfk.losses.SparseCategoricalCrossentropy()(tf.convert_to_tensor([classIdx]),
				predictions)
			# check to see if we are logging the loss value, and if
			# so, display it to our terminal
			if step % 5 == 0:
				print("step: {}, loss: {}...".format(step,
					loss.numpy()))
		# calculate the gradients of loss with respect to the
		# perturbation vector
		gradients = tape.gradient(loss, delta)
		# update the weights, clip the perturbation vector, and
		# update its value
		tfk.optimizers.Adam().apply_gradients([(gradients, delta)])
		delta.assign_add(clip_eps(delta, eps=0.0001))
	# return the perturbation vector
	return delta

test_image = train_images[1]
image = tf.expand_dims(test_image, axis=0)

# create a tensor based off the input image and initialize the
# perturbation vector (we will update this vector via training)
baseImage = tf.constant(image, dtype=tf.float64)
delta = tf.Variable(tf.zeros_like(baseImage), trainable=True
                    )
# generate the perturbation vector to create an adversarial example
deltaUpdated = generate_adversaries(model, baseImage, delta, 0, steps=100)

# create the adversarial example, swap color channels, and save the
# output image to disk
adverImage = (baseImage + deltaUpdated).numpy().squeeze()
adverImage = np.clip(adverImage, 0, 1).astype("float32")

prediction_init = model.predict(baseImage)[0]
prediction_adversarial = model.predict(tf.expand_dims(adverImage, axis=0))[0]

label_init = class_names[np.argmax(prediction_init)]
confidence_init = np.max(prediction_init)
label_adversarial = class_names[np.argmax(prediction_adversarial)]
confidence_adversarial = np.max(prediction_adversarial)

plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(baseImage[0,:,:,:])
plt.xlabel('Image\nGuess: ' + str(label_init) + ' conf.' +str(round(confidence_init,2)))
plt.subplot(1,3,2)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(delta[0,:,:,:]*100)
plt.xlabel('Delta*100')
plt.subplot(1,3,3)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(adverImage)
plt.xlabel('Adversarial\nGuess: ' + str(label_adversarial) + ' conf.' +str(round(confidence_adversarial,2)))
#plt.show()


plt.savefig(impath + 'adversarial')
plt.clf()