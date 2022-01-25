import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import tensorflow  as tf

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