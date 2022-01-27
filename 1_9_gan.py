import warnings

import numpy             as np
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import matplotlib.pyplot as plt
import tensorflow  as tf

from model_builders import get_gan
#from custom_utils import train_and_evaluate


# Load and reformat Fashion MNIST dataset 
fashion_mnist = tfk.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
		   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images[:,:,:,np.newaxis]
train_images = train_images / 255
test_images = test_images[:,:,:,np.newaxis]
test_images =  test_images/ 255

train_labels = tfk.utils.to_categorical(train_labels)
test_labels = tfk.utils.to_categorical(test_labels)

x = np.concatenate((train_images, test_images), axis=0)


def generate_real_samples(dataset, n_samples):
  # choose random instances
  i = np.random.randint(0, dataset.shape[0], n_samples)
  # retrieve selected images
  X = dataset[i] 
  # generate 'real' class labels (1)
  y = np.ones((n_samples, 1)) 

  return X, y 

def generate_latent_points(latent_dim, n_samples):
  
  # generate points in the latent space
  x_input = np.random.randn(latent_dim * n_samples)
  # reshape into a batch of inputs for the network
  x_input = x_input.reshape(n_samples, latent_dim)
  return x_input

def generate_fake_samples(generator, latent_dim, n_samples):
  
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create 'fake' class labels (0)
	y = np.zeros((n_samples, 1))
	return X, y

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
  # plot images
  for i in range(n * n):
  # define subplot
    plt.subplot(n, n, 1 + i)
    # turn off axis
    plt.axis('off')
    # plot raw pixel data
    plt.imshow(examples[i, :, :, 0], cmap='gray_r')
  # save plot to file
  filename = './img_1_9_gan/generated_plot_e%03d.png' % (epoch+1)
  plt.savefig(filename)
  plt.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, 
                          dataset, latent_dim, n_samples=100):
  # prepare real samples
  X_real, y_real = generate_real_samples(dataset, n_samples)
  # evaluate discriminator on real examples
  _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
  # prepare fake examples
  x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
  # evaluate discriminator on fake examples
  _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
  # summarize discriminator performance
  print('>Accuracy real: %.0f%%,fake: %.0f%%'\
     % (acc_real*100, acc_fake*100))
  # save plot
  save_plot(x_fake, epoch)
  # save the generator model tile file
  filename = './models/generator_model_%03d.h5' % (epoch + 1)
  g_model.save(filename)

def train_gan(generator, discriminator, gan, 
              dataset, epochs=10, batch_size=256):
  
  latent_dim = gan.input_shape[1]
  n_inputs = dataset.shape[0] 
  n_batches = n_inputs // batch_size
  half_batch = batch_size // 2
  # manually enumerate epochs
  for i in range(epochs):
    # enumerate batches over the training set
    for j in range(n_batches):
      # get randomly selected 'real' samples
      X_real, y_real = generate_real_samples(dataset, 
                                             half_batch)
      # generate 'fake' examples
      X_fake, y_fake = generate_fake_samples(generator, 
                                             latent_dim, 
                                             half_batch)
      # create training set for the discriminator
      X = np.concatenate((X_real, X_fake), axis=0) 
      y = np.concatenate((y_real, y_fake), axis=0)
      # update discriminator model weights
      d_loss, _ = discriminator.train_on_batch(X, y)
      # prepare points in latent space as input for the generator
      X_gan = generate_latent_points(latent_dim, batch_size)
      # create inverted labels for the fake samples
      y_gan = np.ones((batch_size, 1))
      # update the generator via the discriminator's error
      g_loss = gan.train_on_batch(X_gan, y_gan)
      # summarize loss on this batch
      print('>%d, %d/%d, d=%.3f, g=%.3f' \
        % (i+1, j+1, n_batches, d_loss, g_loss), end='\r')
      # evaluate the model performance, sometimes
    print('')
    if (i+1) % 10 == 0:
      summarize_performance(i, generator, discriminator, 
                            dataset, latent_dim)
  

ep, bs = 100, 1000
  
generator, discriminator, gan = \
  get_gan(latent_dim=10)

generator.summary() 
discriminator.summary() 
gan.summary()
train_gan(generator, discriminator, gan, 
          x, epochs=ep, batch_size=bs)

generator.save('generator.h5')
discriminator.save('discriminator.h5')
gan.save('gan.h5')
