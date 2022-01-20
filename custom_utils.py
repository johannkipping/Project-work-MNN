from time   import time      # For time measuring
import warnings

import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.keras  as tfk


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

    print('Accuracy = ', history.history['acc'][-1])
    print('Validation accuracy = ', history.history['val_acc'][-1])
    print('Time needed for training: ', train_time)
    print('\n')
    
    # plotting possible for layered pictures
    if acc_bool:
        plt.plot(
            history.history['acc'],
            label='acc. ' + model.title + '  end:' + str(round(history.history['acc'][-1],2))
        )
    plt.plot(
        history.history['val_acc'],
        label = 'val. acc. ' + model.title + '  end:' + str(round(history.history['val_acc'][-1],2))
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