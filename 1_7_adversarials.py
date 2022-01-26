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
model = get_final_model(name='final', input_shape = (32,32,3), **model_param_dict)

model.compile(
      optimizer=model.optimizer, 
      loss='categorical_crossentropy',
      metrics=['accuracy']
)

model.load_weights('./models/final_cifar10')

optimizer = tfk.optimizers.Adam(learning_rate=0.0001)
ccloss = tfk.losses.CategoricalCrossentropy()

def generate_adversaries(model, baseImage, delta, classIdx, steps=50):
	for step in range(0, steps):
		with tf.GradientTape() as tape:
			tape.watch(delta)
			adversary = baseImage + delta
			predictions = model(adversary, training=False)
			loss = -ccloss(tf.convert_to_tensor([classIdx]),predictions)
			if step % 10 == 0:
				print("step: {}, loss: {}...".format(step,
					loss.numpy()))

		gradients = tape.gradient(loss, delta)

		optimizer.apply_gradients([(gradients, delta)])
		eps = 0.0001
		delta_clipped = tf.clip_by_value(delta, clip_value_min=-eps, clip_value_max=eps)
		delta.assign_add(delta_clipped)
	return delta

test_idx = 2
test_image = train_images[test_idx]
image = tf.expand_dims(test_image, axis=0)

baseImage = tf.constant(image, dtype=tf.float64)
initializer = tf.random_normal_initializer()
delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

deltaUpdated = generate_adversaries(model, baseImage, delta, train_labels[test_idx], steps=50)

adverImage = (baseImage + deltaUpdated)
adverImage = np.clip(adverImage, 0, 1).astype("float32")

prediction_init = model.predict(baseImage)[0]
prediction_adversarial = model.predict(adverImage)[0]

label_init = class_names[np.argmax(prediction_init)]
confidence_init = np.max(prediction_init)
label_adversarial = class_names[np.argmax(prediction_adversarial)]
confidence_adversarial = np.max(prediction_adversarial)

plt.figure()
plt.subplot(1,3,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(baseImage[0,:,:,:])
plt.xlabel('Image\nGuess: ' + str(label_init) + '\nConfidence: ' +str(round(confidence_init,2)))
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
plt.imshow(adverImage[0,:,:,:])
plt.xlabel('Adversarial\nGuess: ' + str(label_adversarial) + '\nConfidence: ' +str(round(confidence_adversarial,2)))
#plt.show()


plt.savefig(impath + 'adversarial')
plt.clf()