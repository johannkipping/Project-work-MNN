import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import tensorflow  as tf

class InitModel(tfk.Model):
    """Model from Exercise 4 of sheet 6"""
    def __init__(
            self,
            name='init_model',
            num_classes=10,
            activation='relu',
            initializer='he_uniform',
            eta=0.001
        ):
        super(InitModel, self).__init__(
            name=name
        )
        self.title = name
        self.num_classes = num_classes
        self.learning_rate = eta
        self.optimizer = tf.train.AdamOptimizer(eta)

        # layers of the network
        self.conv2d = tfk.layers.Conv2D(
            32,
            (3,3),
            activation=activation,
            kernel_initializer=initializer
        )

        self.max_pooling2d = tfk.layers.MaxPool2D(pool_size=(2,2))
        self.conv2d_1 = tfk.layers.Conv2D(
            64,
            (3,3),
            activation=activation,
            kernel_initializer=initializer
        )

        self.max_pooling2d_1 = tfk.layers.MaxPool2D(pool_size=(2,2))

        self.flatten = tfk.layers.Flatten()

        self.dense = tfk.layers.Dense(
            2048,
            activation=activation,
            kernel_initializer=initializer
        )

        self.dense_1 = tfk.layers.Dense(
            10,
            activation='softmax',
            kernel_initializer=initializer
        )

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.conv2d(inputs)
        x = self.max_pooling2d(x)
        x = self.conv2d_1(x)
        x = self.max_pooling2d_1(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.dense_1(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
    

class WideModel(tfk.Model):
    """Model from Exercise 4 of sheet 6"""
    def __init__(
            self,
            name='wide_model',
            num_classes=10,
            activation='relu',
            initializer='he_uniform',
            eta=0.001
        ):
        super(WideModel, self).__init__(
            name=name
        )
        self.title = name
        self.num_classes = num_classes
        self.learning_rate = eta
        self.optimizer = tf.train.AdamOptimizer(eta)

        # layers of the network
        self.conv2d = tfk.layers.Conv2D(
            128,
            (3,3),
            activation=activation,
            kernel_initializer=initializer
        )

        self.max_pooling2d = tfk.layers.MaxPool2D(pool_size=(2,2))

        self.conv2d_1 = tfk.layers.Conv2D(
            512,
            (3,3),
            activation=activation,
            kernel_initializer=initializer
        )

        self.max_pooling2d_1 = tfk.layers.MaxPool2D(pool_size=(2,2))

        self.flatten = tfk.layers.Flatten()

        self.dense = tfk.layers.Dense(
            4096,
            activation=activation,
            kernel_initializer=initializer
        )

        self.dense_1 = tfk.layers.Dense(
            10,
            activation='softmax',
            kernel_initializer=initializer
        )

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.conv2d(inputs)
        x = self.max_pooling2d(x)
        x = self.conv2d_1(x)
        x = self.max_pooling2d_1(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.dense_1(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
        
        
class FlatModel(tfk.Model):
    """Model from Exercise 4 of sheet 6"""
    def __init__(
            self,
            name='flat_model',
            num_classes=10,
            activation='relu',
            initializer='he_uniform',
            eta=0.001,
            neurons=256
        ):
        super(FlatModel, self).__init__(
            name=name
        )
        self.title = name
        self.num_classes = num_classes
        self.learning_rate = eta
        self.optimizer = tf.train.AdamOptimizer(eta)

        # layers of the network
        self.flatten = tfk.layers.Flatten()

        self.dense = tfk.layers.Dense(
            neurons,
            activation=activation,
            kernel_initializer=initializer
        )

        self.dense_1 = tfk.layers.Dense(
            10,
            activation='softmax',
            kernel_initializer=initializer
        )

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.flatten(inputs)
        x = self.dense(x)
        return self.dense_1(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
    
    

class DeeperModel(tfk.Model):
    """Model from Exercise 4 of sheet 6"""
    def __init__(
            self,
            name='init_model',
            num_classes=10,
            activation='relu',
            initializer='he_uniform',
            eta=0.001
        ):
        super(DeeperModel, self).__init__(
            name=name
        )
        self.title = name
        self.num_classes = num_classes
        self.learning_rate = eta
        self.optimizer = tf.train.AdamOptimizer(eta)

        # layers of the network
        self.conv2d1 = tfk.layers.Conv2D(
            32,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.conv2d2 = tfk.layers.Conv2D(
            32,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )

        self.max_pooling2d = tfk.layers.MaxPool2D(pool_size=(2,2))

        self.conv2d_11 = tfk.layers.Conv2D(
            64,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.conv2d_12 = tfk.layers.Conv2D(
            64,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )

        self.max_pooling2d_1 = tfk.layers.MaxPool2D(pool_size=(2,2))
        
        self.conv2d_21 = tfk.layers.Conv2D(
            128,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.conv2d_22 = tfk.layers.Conv2D(
            128,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )

        self.max_pooling2d_2 = tfk.layers.MaxPool2D(pool_size=(2,2))
        
        self.flatten = tfk.layers.Flatten()

        self.dense = tfk.layers.Dense(
            1028,
            activation=activation,
            kernel_initializer=initializer
        )

        self.dense_1 = tfk.layers.Dense(
            10,
            activation='softmax',
            kernel_initializer=initializer
        )

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.conv2d1(inputs)
        x = self.conv2d2(x)
        x = self.max_pooling2d(x)
        x = self.conv2d_11(x)
        x = self.conv2d_12(x)
        x = self.max_pooling2d_1(x)
        x = self.conv2d_21(x)
        x = self.conv2d_22(x)
        x = self.max_pooling2d_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.dense_1(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
        
        
class DropoutModel(tfk.Model):
    """Model from Exercise 4 of sheet 6"""
    def __init__(
            self,
            name='init_model',
            num_classes=10,
            activation='relu',
            initializer='he_uniform',
            eta=0.001,
            drop_prob=0.2
        ):
        super(DropoutModel, self).__init__(
            name=name
        )
        self.title = name
        self.num_classes = num_classes
        self.learning_rate = eta
        self.optimizer = tf.train.AdamOptimizer(eta)

        # layers of the network
        self.conv2d1 = tfk.layers.Conv2D(
            32,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.conv2d2 = tfk.layers.Conv2D(
            32,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )

        self.max_pooling2d = tfk.layers.MaxPool2D(pool_size=(2,2))
        self.dropout = tfk.layers.Dropout(drop_prob)

        self.conv2d_11 = tfk.layers.Conv2D(
            64,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.conv2d_12 = tfk.layers.Conv2D(
            64,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )

        self.max_pooling2d_1 = tfk.layers.MaxPool2D(pool_size=(2,2))
        self.dropout_1 = tfk.layers.Dropout(drop_prob)
        
        self.conv2d_21 = tfk.layers.Conv2D(
            128,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.conv2d_22 = tfk.layers.Conv2D(
            128,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )

        self.max_pooling2d_2 = tfk.layers.MaxPool2D(pool_size=(2,2))
        self.dropout_2 = tfk.layers.Dropout(drop_prob)
        
        self.flatten = tfk.layers.Flatten()

        self.dense = tfk.layers.Dense(
            1028,
            activation=activation,
            kernel_initializer=initializer
        )
        self.dropout_3 = tfk.layers.Dropout(drop_prob)

        self.dense_1 = tfk.layers.Dense(
            10,
            activation='softmax',
            kernel_initializer=initializer
        )

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.conv2d1(inputs)
        x = self.conv2d2(x)
        x = self.max_pooling2d(x)
        x = self.dropout(x)
        x = self.conv2d_11(x)
        x = self.conv2d_12(x)
        x = self.max_pooling2d_1(x)
        x = self.dropout_1(x)
        x = self.conv2d_21(x)
        x = self.conv2d_22(x)
        x = self.max_pooling2d_2(x)
        x = self.dropout_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout_3(x)
        return self.dense_1(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
    
    
class BatchnormModel_baseline(tfk.Model):
    """Model from Exercise 4 of sheet 6"""
    def __init__(
            self,
            name='init_model',
            num_classes=10,
            activation='relu',
            initializer='he_uniform',
            eta=0.001
        ):
        super(BatchnormModel, self).__init__(
            name=name
        )
        self.title = name
        self.num_classes = num_classes
        self.learning_rate = eta
        self.optimizer = tf.train.AdamOptimizer(eta)

        # layers of the network
        self.conv2d = tfk.layers.Conv2D(
            32,
            (3,3),
            activation=activation,
            kernel_initializer=initializer
        )
        self.batchnorm = tfk.layers.BatchNormalization()
        self.max_pooling2d = tfk.layers.MaxPool2D(pool_size=(2,2))
        self.conv2d_1 = tfk.layers.Conv2D(
            64,
            (3,3),
            activation=activation,
            kernel_initializer=initializer
        )
        self.batchnorm_1 = tfk.layers.BatchNormalization()
        self.max_pooling2d_1 = tfk.layers.MaxPool2D(pool_size=(2,2))
        self.flatten = tfk.layers.Flatten()
        self.dense = tfk.layers.Dense(
            2048,
            activation=activation,
            kernel_initializer=initializer
        )
        self.dense_1 = tfk.layers.Dense(
            10,
            activation='softmax',
            kernel_initializer=initializer
        )

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.conv2d(inputs)
        x = self.max_pooling2d(x)
        x = self.batchnorm(x)
        x = self.conv2d_1(x)
        x = self.max_pooling2d_1(x)
        x = self.batchnorm_1(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.dense_1(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

   
class BatchnormModel(tfk.Model):
    """Model from Exercise 4 of sheet 6"""
    def __init__(
            self,
            name='init_model',
            num_classes=10,
            activation='relu',
            initializer='he_uniform',
            eta=0.001
        ):
        super(BatchnormModel, self).__init__(
            name=name
        )
        self.title = name
        self.num_classes = num_classes
        self.learning_rate = eta
        self.optimizer = tf.train.AdamOptimizer(eta)

        # layers of the network
        self.conv2d1 = tfk.layers.Conv2D(
            32,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.batchnorm = tfk.layers.BatchNormalization()
        self.conv2d2 = tfk.layers.Conv2D(
            32,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.batchnorm_1 = tfk.layers.BatchNormalization()
        self.max_pooling2d = tfk.layers.MaxPool2D(pool_size=(2,2))

        self.conv2d_11 = tfk.layers.Conv2D(
            64,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.batchnorm_2 = tfk.layers.BatchNormalization()
        self.conv2d_12 = tfk.layers.Conv2D(
            64,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.batchnorm_3 = tfk.layers.BatchNormalization()

        self.max_pooling2d_1 = tfk.layers.MaxPool2D(pool_size=(2,2))
        
        self.conv2d_21 = tfk.layers.Conv2D(
            128,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.batchnorm_4 = tfk.layers.BatchNormalization()
        self.conv2d_22 = tfk.layers.Conv2D(
            128,
            (3,3),
            activation=activation,
            kernel_initializer=initializer,
            padding='same'
        )
        self.batchnorm_5 = tfk.layers.BatchNormalization()
        
        self.max_pooling2d_2 = tfk.layers.MaxPool2D(pool_size=(2,2))
        
        self.flatten = tfk.layers.Flatten()

        self.dense = tfk.layers.Dense(
            1028,
            activation=activation,
            kernel_initializer=initializer
        )

        self.dense_1 = tfk.layers.Dense(
            10,
            activation='softmax',
            kernel_initializer=initializer
        )

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.conv2d1(inputs)
        x = self.batchnorm(x)
        x = self.conv2d2(x)
        x = self.batchnorm_1(x)
        x = self.max_pooling2d(x)
        x = self.conv2d_11(x)
        x = self.batchnorm_2(x)
        x = self.conv2d_12(x)
        x = self.batchnorm_3(x)
        x = self.max_pooling2d_1(x)
        x = self.conv2d_21(x)
        x = self.batchnorm_4(x)
        x = self.conv2d_22(x)
        x = self.batchnorm_5(x)
        x = self.max_pooling2d_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.dense_1(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
    
    
class Autoencoder(tfk.Model):
    def __init__(
            self,
            latent_dim=3,
            name='autoencoder',
            activation='relu',
            initializer='he_uniform',
            eta=0.001
        ):
        super(Autoencoder, self).__init__()
        self.title = name
        self.learning_rate = eta
    
        if tf.__version__[0]=='1':
            self.optimizer = tf.train.AdamOptimizer(eta)
        else:
            self.optimizer = tf.optimizers.Adam(eta)
        
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            tfk.layers.Flatten(),
            tfk.layers.Dense(128, activation=activation),
            tfk.layers.Dense(latent_dim, activation=activation)
        ])
        self.decoder = tf.keras.Sequential([
            tfk.layers.Dense(128, activation=activation),
            tfk.layers.Dense(784, activation='sigmoid'),
            tfk.layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
