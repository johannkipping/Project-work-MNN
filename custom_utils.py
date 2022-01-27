from time   import time      # For time measuring
import warnings

import numpy as np
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk
import tensorflow  as tf


def train_and_evaluate(
        model,
        acc_bool=False,
        info_str='',
        batch_size=128,
        epochs=10,
        train_images=None,
        train_labels=None,
        test_images=None,
        test_labels=None,
    ):
    start = time()
    history = model.fit(
          train_images,
          train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test_images, test_labels)
    )
    train_time = time() - start
    
    #model.save_weights('./' + 'weights_' + folder_name + '/' + model.title)

    model.summary()

    # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    
    if tf.__version__[0]=='1':
        key_acc = 'acc'
        key_val_acc = 'val_acc'
    else:
        key_acc = 'accuracy'
        key_val_acc = 'val_accuracy'
    

    print('Accuracy = ', history.history[key_acc][-1])
    print('Validation accuracy = ', history.history[key_val_acc][-1])
    print('Time needed for training: ', train_time)
    print('\n')
    
    # plotting possible for layered pictures
    if acc_bool:
        plt.plot(
            history.history[key_acc],
            label='acc. ' + model.title + '  end:' + str(round(history.history[key_acc][-1],2))
        )
    plt.plot(
        history.history[key_val_acc],
        label = 'val. acc. ' + model.title + '  end:' + str(round(history.history[key_val_acc][-1],2))
    )
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    
    info_str += (
        model.title 
        + ' bs:' + str(batch_size)
        + ' ep:' + str(epochs)
        + ' eta:' + str(model.learning_rate)
        + ' t:' + str(round(train_time))
        + 's\n'
    )
    
    plt.title(info_str)
    
    return info_str

def plot_mnist_latent_space(x, y, class_names, encoder, name, dim):
        
    labels = y
    tf_encoded = encoder.predict(x)

    fig = plt.figure()
    if dim <3:
        plt.scatter(tf_encoded[:, 0],
                    tf_encoded[:, 1],
                    s=1, c=labels, cmap='tab10')
        plt.title('Latent space of Fashion-MNIST AE', fontsize=10)
        plt.xlabel('x')
        plt.ylabel('y')
        cbar = plt.colorbar()
        cbar.set_ticks(ticks=np.linspace(0.5,8.5,10))
        cbar.set_ticklabels(class_names)
        
        plt.savefig(name, bbox_inches='tight')
    else:
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(
                    tf_encoded[:, 0],
                    tf_encoded[:, 1],
                    tf_encoded[:, 2],
                    s=1, c=labels, cmap='tab10')
        ax.set_title('Latent space of Fashion-MNIST AE', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        cbar = fig.colorbar(sc)
        cbar.set_ticks(ticks=np.linspace(0.5,8.5,10))
        cbar.set_ticklabels(class_names)
    
        plt.show()
 
 
def plot_latent_space_interpolation(encoder, data, labels, decoder, name,
                                    scale_x=1.0, scale_y=1.0,
                                    n=30, m=30, figsize=(15,15)):
    # display a n*n 2D manifold of digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * m))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    correction = scale_x / (m - 1)
    grid_x = np.linspace(0, scale_x, m)
    grid_y = np.linspace(0, scale_y, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample).clip(min=0.0, max=1.0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    encoded = encoder.predict(data)
    plt.figure(figsize=figsize)
    scale_x += correction
    scale_y += correction
    axis = [0, scale_x, 0, scale_y]
    plt.scatter(encoded[:, 0],
                encoded[:, 1],
                s=16, c=labels, cmap='tab10', alpha=.1) # alpha?

    plt.imshow(figure, extent=axis, cmap='gray_r')
    plt.axis(axis)
    plt.savefig(name, bbox_inches='tight')
