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
        
        self.conv2d_2 = tfk.layers.Conv2D(
            64,
            (3,3),
            activation=activation,
            kernel_initializer=initializer
        )

        self.max_pooling2d_2 = tfk.layers.MaxPool2D(pool_size=(2,2))
        
        self.flatten = tfk.layers.Flatten()

        self.dense = tfk.layers.Dense(
            2048,
            activation=activation,
            kernel_initializer=initializer
        )
        
        self.dense_1 = tfk.layers.Dense(
            2048,
            activation=activation,
            kernel_initializer=initializer
        )

        self.dense_2 = tfk.layers.Dense(
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
        x = self.conv2d_2(inputs)
        x = self.max_pooling2d_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dense_1(x)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)