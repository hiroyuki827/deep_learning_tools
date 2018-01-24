"""Test code on Travis.CI"""
# For Travis
import os

import matplotlib as mpl
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import dlt

if __name__ == '__main__':


    # ---------------------------------------------------------
    # Load and preprocess data
    # ---------------------------------------------------------
    data = dlt.fashion_mnist.load_data()

    # plot some example images
    dlt.utils.plot_examples(data, fname='examples.png')

    # preprocess the data in a suitable way
    # reshape the image matrices to vectors
    # RGB 255 = white, 0 = black
    X_train = data.train_images.reshape([-1, 28, 28, 1])
    X_test = data.test_images.reshape([-1, 28, 28, 1])
    print('%i training samples' % X_train.shape[0])
    print('%i test samples' % X_test.shape[0])
    print(X_train.shape)

    # convert integer RGB values (0-255) to float values (0-1)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # convert class labels to one-hot encodings
    Y_train = to_categorical(data.train_labels, 10)
    Y_test = to_categorical(data.test_labels, 10)

    # Plot data distribution for Y_train
    dlt.utils.plot_distribution_data(Y=data.train_labels,
                                     dataset_name='y_train',
                                     classes=data.classes,
                                     fname='dist_train.png')
    # ----------------------------------------------------------
    # Model and training
    # ----------------------------------------------------------

    num_classes = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    fit = model.fit(X_train, Y_train,
                    batch_size=128,
                    epochs=12,
                    verbose=1,
                    validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test,
                           verbose=0
                           )
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # ----------------------------------------------
    # Some plots
    # ----------------------------------------------

    # model.save(os.path.join(folder, 'my_model.h5'))

    # predicted probabilities for the test set
    Yp = model.predict(X_test)
    yp = np.argmax(Yp, axis=1)

    # plot some test images along with the prediction
    for i in range(10):
        dlt.utils.plot_prediction(
            Yp[i],
            data.test_images[i],
            data.test_labels[i],
            data.classes,
            fname='test-%i.png' % i)

    # plot the confusion matrix
    dlt.utils.plot_confusion_matrix(data.test_labels, yp, data.classes,
                                    title='confusion matrix',
                                    fname='confusion matrix.png')

    # plot the loss and accuracy graph
    dlt.utils.plot_loss_and_accuracy(fit,  # model.fitのインスタンス
                                     fname='loss_and_accuracy_graph.png'  # 保存するファイル名とパス
                                     )
