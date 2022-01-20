from time   import time      # For time measuring
import warnings

import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk


def train_and_evaluate(
        model,
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

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('Test accuracy = ', test_acc)
    print('Time needed for training: ', train_time)

    plt.plot(history.history['acc'], label='acc_'+model.title)
    plt.plot(history.history['val_acc'], label = 'val_accuracy'+model.title)
    plt.title(
        model.title + ' with epochs ' + str(epochs) + '\n' 
        + 'Batch size: ' + str(batch_size) 
        + '  learning rate: ' + str(model.learning_rate) + '\n'
        + 'Time: ' + str(round(train_time,1))
        + '  acc.: ' + str(round(history.history['acc'][-1],2))
        + '  test acc.: ' + str(round(test_acc,2))
    )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    
    print('\n\n')