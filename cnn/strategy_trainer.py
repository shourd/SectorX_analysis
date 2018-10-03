import numpy as np
from config import Settings
import pandas as pd
from cnn.ssd_loader import ssd_loader
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from os import environ, path, makedirs


def ssd_trainer(x_data, actions):

    iteration_name = 'model'

    actions_type = list(actions.TYPE)
    res_list = []
    for command in actions_type:
        if command == 'HDG': res = 0
        if command == 'SPD': res = 1
        if command == 'DCT': res = 2
        res_list.append(res)
    print(res_list)
    y_data = keras.utils.to_categorical(res_list, 3)
    print(y_data)

    """ SPLIT TRAIN AND VALIDATION DATA """
    train_length = int(settings.train_val_ratio * len(x_data))
    x_train = x_data[0:train_length, :, :, :]
    x_val = x_data[train_length:, :, :, :]

    y_train = y_data[0:train_length, :]
    y_val = y_data[train_length:, :]

    data_length = len(x_train) + len(x_val)
    print('Amount of SSDs: {} ({} train / {} val)'.format(data_length, len(x_train), len(x_val)))

    """ CREATING THE NEURAL NETWORK """
    model = Sequential()
    input_shape = (settings.ssd_import_size[0], settings.ssd_import_size[1], 3)
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    # BASELINE ARCHITECTURE
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    if settings.ssd_import_size[0] > 32:
        model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    if settings.ssd_import_size[0] > 16:
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening and FC
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(settings.num_classes, activation='softmax'))
    model.summary()

    # Output model structure to disk
    if not path.exists(settings.output_dir):
        print('Output directory created')
        makedirs(settings.output_dir)

    plot_model(model, to_file='structure.png', show_shapes=True, show_layer_names=False)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # sgd = keras.optimizers.SGD(lr=0.001)
    # model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

    """ CALLBACKS """

    # HISTORY
    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()

    # CHECKPOINTS
    filepath = settings.output_dir + '/' + iteration_name + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    if settings.save_model:
        callbacks_list = [checkpoint, history]
    else:
        callbacks_list = [history]

    # For debugging purposes. Exports augmented image data
    export_dir = None
    # if export_dir is not None:
    #     try:
    #         makedirs(export_dir)
    #     except FileExistsError:
    #         shutil.rmtree(export_dir)
    #         makedirs(export_dir)

    train_datagen = ImageDataGenerator(
        rotation_range=settings.rotation_range,
        fill_mode='constant',
        cval=1,  # fill with white pixels [0;1]
    )

    train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=settings.batch_size,
        save_to_dir=export_dir,
        save_prefix='aug',
        save_format='png')

    # measure training time
    # start_time = time.time()

    # start training
    model.fit_generator(
        train_generator,
        steps_per_epoch=settings.steps_per_epoch,  # len(x_train) / settings.batch_size,
        epochs=settings.epochs,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=callbacks_list)

    score = model.evaluate(x_val, y_val, verbose=0)
    test_loss = round(score[0], 3)
    test_accuracy = round(score[1], 3)
    # train_time = int(time.time() - start_time)


if __name__ == "__main__":
    settings = Settings
    ssd_stack, actions = ssd_loader(settings)
    ssd_trainer(ssd_stack, actions)