import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import tensorflow  as tf

#import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from tensorflow import keras
from tensorflow.keras.layers import Input,\
   Conv2D, Conv2DTranspose, Dense,\
   LeakyReLU, Dropout, Flatten, Reshape 

def get_final_model(
        name='final_model',
        num_classes=10,
        activation='relu',
        initializer='he_uniform',
        eta=0.001,
        input_shape = (28,28,1),
        drop_prob=0.25
    ):
    """Final model with everything combined"""
    model = tfk.Sequential()
    model.title = name
    model.num_classes = num_classes
    model.learning_rate = eta
    
    if tf.__version__[0]=='1':
        model.optimizer = tf.train.AdamOptimizer(eta)
    else:
        model.optimizer = tf.optimizers.Adam(eta)

    # layers of the network
    model.add(tfk.layers.Conv2D(
        32,
        (3,3),
        input_shape=input_shape,
        activation=activation,
        kernel_initializer=initializer,
        padding='same'
    ))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Conv2D(
        32,
        (3,3),
        activation=activation,
        kernel_initializer=initializer,
        padding='same'
    ))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tfk.layers.Dropout(drop_prob))

    model.add(tfk.layers.Conv2D(
        64,
        (3,3),
        activation=activation,
        kernel_initializer=initializer,
        padding='same'
    ))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Conv2D(
        64,
        (3,3),
        activation=activation,
        kernel_initializer=initializer,
        padding='same'
    ))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tfk.layers.Dropout(drop_prob))
    
    model.add(tfk.layers.Conv2D(
        128,
        (3,3),
        activation=activation,
        kernel_initializer=initializer,
        padding='same'
    ))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Conv2D(
        128,
        (3,3),
        activation=activation,
        kernel_initializer=initializer,
        padding='same'
    ))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tfk.layers.Dropout(drop_prob))
    
    model.add(tfk.layers.Flatten())

    model.add(tfk.layers.Dense(
        1028,
        activation=activation,
        kernel_initializer=initializer
    ))
    model.add(tfk.layers.Dropout(drop_prob))

    model.add(tfk.layers.Dense(
        10,
        activation='softmax',
        kernel_initializer=initializer
    ))
    
    return model


def get_gan(latent_dim):
    input_shape = (latent_dim,)
    opt = keras.optimizers.Adam(learning_rate=.0002, beta_1=.5)
    # generative part
    generator = keras.Sequential()
    generator.add(Input(shape=input_shape))
    generator.add(Dense(128*7*7))
    generator.add(LeakyReLU(alpha=0.2)) 
    generator.add(Reshape((7, 7, 128))) 
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    generator.compile(optimizer=opt, loss='binary_crossentropy')
    # discriminative part
    discriminator = keras.Sequential()
    discriminator.add(Input(shape=(28,28,1)))
    discriminator.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.4)) 
    discriminator.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.4))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))

    discriminator.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Set weights as not trainable. 
    # This does not effect the discriminator model 
    # directly (since it was compiled with trainable
    # weights before), but the whole GAN 
    # (since it is compiled afterwards).
    discriminator.trainable=False
    # Define GAN
    gan = keras.Sequential()
    gan.add(Input(shape=input_shape))
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(optimizer=opt, loss='binary_crossentropy')
    

    return generator, discriminator, gan 