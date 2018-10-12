import numpy as np
from config import Settings
import pandas as pd
from ssd_loader import ssd_loader
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Activation
from keras import backend as K
from keras.applications import VGG16
from os import environ, path, makedirs
import pickle
import time
import matplotlib.pyplot as plt
from PIL import Image


def ssd_trainer(all_data):
    iteration_name = 'iteration_name'

    """ PREPARE TRAINING SET """
    x_data, y_data = prepare_training_set(all_data['ssd_images'], all_data['commands'])

    """ SPLIT TRAIN AND VALIDATION DATA """
    x_train, y_train, x_val, y_val = split_data(x_data, y_data, settings)

    """ CREATE MODEL """
    # model, layer = create_model(settings)
    model, layer = create_pretrained_model(settings)

    """ CALLBACKS """

    # HISTORY
    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()
    tensor_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32,
                                                        write_graph=True, write_grads=True,
                                                        write_images=True, embeddings_freq=0,
                                                        embeddings_layer_names=None,
                                                        embeddings_metadata=None, embeddings_data=None)

    # CHECKPOINTS
    filepath = settings.output_dir + '/' + iteration_name + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    if settings.save_model:
        callbacks_list = [checkpoint, history]
    else:
        callbacks_list = [history, tensor_board_callback]

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
    start_time = time.time()

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
    train_time = int(time.time() - start_time)

    print('Test accuracy:', test_accuracy)

    visualize_layer(x_train, model, layer)


def prepare_training_set(ssd_data, command_data):
    """ FILTER COMMANDS """
    participant_ids = ['P1']  # ['P7']
    run_ids = ['R1']#'all'
    command_types = ['SPD', 'HDG']
    settings.num_classes = len(command_types)

    command_data = command_data[command_data.ssd_id != 'N/A']
    command_data.reset_index(inplace=True)
    command_data = command_data.sort_values(by=['ssd_id'])
    print(command_data.to_string())
      # create an ID for all actions
    # command_data = command_data[command_data.PARTICIPANT_ID in participant_ids]

    if participant_ids is not 'all':
        command_data = command_data[command_data.PARTICIPANT_ID.isin(participant_ids)]
    if run_ids is not 'all':
        command_data = command_data[command_data.RUN_ID.isin(run_ids)]
    if command_types is not 'all':
        command_data = command_data[command_data.TYPE.isin(command_types)]


    print(command_data.to_string())

    # filter ssds based on remaining actions
    actions_ids = command_data.index.unique()
    x_data = ssd_data[actions_ids, :, :, :]

    print('Speed commands:', len(command_data[command_data.TYPE == 'SPD']))

    res_list = []
    for command in list(command_data.TYPE):
        if command == 'HDG': res = 0
        elif command == 'SPD': res = 1
        elif command == 'DCT': res = 2
        elif command == 'TOC': res = 3
        else:
            print('ERROR: Command type not recognized')
            break
        res_list.append(res)

    y_data = keras.utils.to_categorical(res_list, settings.num_classes)

    return x_data, y_data


def split_data(x_data, y_data, settings):
    train_length = int(settings.train_val_ratio * len(x_data))
    x_train = x_data[0:train_length, :, :, :]
    x_val = x_data[train_length:, :, :, :]

    y_train = y_data[0:train_length, :]
    y_val = y_data[train_length:, :]

    data_length = len(x_train) + len(x_val)
    print('Number of SSDs to network: {} ({} train / {} val)'.format(data_length, len(x_train), len(x_val)))

    return x_train, y_train, x_val, y_val


def create_model(settings):
    """ CREATING THE NEURAL NETWORK """
    model = Sequential()
    input_shape = settings.ssd_shape
    # input_shape = (settings.ssd_import_size[0], settings.ssd_import_size[1], 3)
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    # BASELINE ARCHITECTURE
    model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1)))
    convout = Activation('relu')
    model.add(convout)
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(10, 10), strides=(2, 2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1)))
    # convout2 = Activation('relu')
    # model.add(convout2)

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

    return model, convout

def create_pretrained_model(settings):


    # Load the VGG model
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=settings.ssd_shape)
    vgg_conv.summary()

    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = Sequential()
    model.add(vgg_conv) # Add the vgg convolutional base model

    convout = model.layers[0].layers[1]  # take the first layer

    # Add new layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    if settings.use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(settings.num_classes, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    # Output model structure to disk
    if not path.exists(settings.output_dir):
        print('Output directory created')
        makedirs(settings.output_dir)

    plot_model(model, to_file='structure_pretrained.png', show_shapes=True, show_layer_names=False)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model, convout

def visualize_layer(x_train, model, layer):
    # choose any image to want by specifying the index
    img_to_visualize = x_train[1]
    # Keras requires the image to be in 4D
    # So we add an extra dimension to it.
    img_to_visualize = np.expand_dims(img_to_visualize, axis=0)

    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print('Shape of conv:', convolutions.shape)

    n = convolutions.shape[2]
    n = int(np.ceil(np.sqrt(n)))

    convolutions = convolutions.transpose((1, 0, 2))
    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12, 8))
    for i in range(convolutions.shape[2]):
        ax = fig.add_subplot(n, n, i + 1)
        ax.imshow(convolutions[:,:,i], cmap='gray')
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(convolutions[:,:,1], cmap='gray')
    plt.show()


if __name__ == "__main__":
    settings = Settings()
    try:
        all_data = pickle.load(open(settings.data_folder + 'all_dataframes_3.p', "rb"))
        print('Data loaded from pickle')
    except FileNotFoundError:
        print('Start importing data from image files')
        all_data = ssd_loader(settings)

    ssd_trainer(all_data)
