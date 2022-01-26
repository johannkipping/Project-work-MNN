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


def plot_latent_space_autoencoder_mnist(
		x_test,
		y_test,
		encoder,
		decoder, 
		autoencoder,
		dim,
		plot_name
    ):

	labels = y_test
	W1_pre, W2_pre = autoencoder.get_weights()
	W1, W2 = np.array(W1_pre), np.array(W2_pre)
	tf_encoded = x_test @ W1
	tf_encoded2 = encoder.predict(x_test)
	print(np.linalg.norm(tf_encoded-tf_encoded2))
	fig = plt.figure(figsize=(5.5, 4))
	if dim == 2:
		ax = fig.add_subplot()
		sc = ax.scatter(tf_encoded2[:, 0],
						tf_encoded2[:, 1],
						s=1, c=labels, cmap='tab10')
		ax.set_title('Latent space of MNIST linear AE', fontsize=10)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
	elif dim == 3:
		ax = fig.add_subplot(projection='3d')
		sc = ax.scatter(tf_encoded2[:, 0],
						tf_encoded2[:, 1],
						tf_encoded2[:, 2],
						s=1, c=labels, cmap='tab10')
		ax.set_title('Latent space of MNIST linear AE', fontsize=10)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
	#plt.axis('square')
	cbar = fig.colorbar(sc)
	cbar.set_ticks(ticks=np.linspace(0.5,8.5,10))
	cbar.set_ticklabels(['0','1','2','3','4','5','6','7','8','9'])
	plt.savefig(plot_name+'.pdf') 