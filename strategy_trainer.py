import pickle
import time
from os import path, makedirs

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.core import Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

import target_data_preparation
from config import settings
from confusion_matrix_script import get_confusion_metrics
from ssd_loader import ssd_loader


def ssd_trainer(all_data, participant_ids):
    K.clear_session()

    """ PREPARE TRAINING SET """
    x_data, y_data = prepare_training_set(all_data['ssd_images'], all_data['commands'], participant_ids)

    """ SPLIT TRAIN AND VALIDATION DATA """
    # x_train, y_train, x_val, y_val = split_data(x_data, y_data) # own function.
    try:
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 1-settings.train_val_ratio,
                                                      shuffle=True, stratify=y_data)
    except ValueError:
        print('---------------------------------- ERROR ----------------------------------')
        print('Only one data sample of one class avaialbe. Not enough for stratified class distribution.')

    """ CREATE MODEL """
    # model, layer = create_model(settings)
    model, layer = create_model(settings.iteration_name)

    """ CALLBACKS """

    # HISTORY
    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))
            # classification_metrics(model, x_val, y_val)


    history = AccuracyHistory()

    # TENSORBOARD
    log_dir = settings.output_dir + '/logs/' + settings.iteration_name
    tensor_board_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, batch_size=32,
                                                        write_graph=False, write_grads=True,
                                                        write_images=False, embeddings_freq=0,
                                                        embeddings_layer_names=None,
                                                        embeddings_metadata=None, embeddings_data=None)

    # CHECKPOINTS
    weights_folder = settings.output_dir + '/weights/'
    makedirs(weights_folder, exist_ok=True)
    checkpoint = ModelCheckpoint(weights_folder + settings.iteration_name + '.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # CSV Outputs
    csv_logger = CSVLogger(settings.output_dir + '/log_{}.csv'.format(settings.iteration_name), append=True, separator=',')


    # Classification metrics
    class Metrics(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self._data = []
            self.epoch_no = 1

        def on_epoch_end(self, batch, logs={}):
            y_pred = model.predict_classes(x_val)
            y_true = np.argmax(y_val, axis=1)
            informedness, F1_score, MCC = get_confusion_metrics(y_true, y_pred, self.epoch_no)
            print('Informedness: {}; MCC: {}'.format(informedness, MCC))

            self._data.append({
                'epoch': self.epoch_no,
                # 'iteration_name': settings.iteration_name,
                'participant': settings.current_participant,
                'target_type': settings.target_type,
                'experiment_name':settings.experiment_name,
                'val_acc': logs.get('val_acc'),
                'val_informedness': informedness,
                'val_F1_score': F1_score,
                'MCC': MCC
            })

            self.epoch_no += 1

            return

        def get_data(self):
            return self._data

    confusion_metrics = Metrics()

    callbacks_list = [history, tensor_board_callback, confusion_metrics]

    if settings.save_model:
        callbacks_list.append(checkpoint)

    if settings.csv_logger:
        callbacks_list.append(csv_logger)

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

    """ CLOSING """

    score = model.evaluate(x_val, y_val, verbose=0)
    # test_loss = round(score[0], 3)
    test_accuracy = round(score[1], 3)
    train_time = int(time.time() - start_time)

    # print('Test accuracy:', test_accuracy)
    print('Train time: {} min'.format(round(train_time/60),1))

    # visualize_layer(x_train, model, layer)

    confusion_metrics_dict = confusion_metrics.get_data()
    confusion_metrics_df = pd.DataFrame.from_dict(confusion_metrics_dict)
    print(confusion_metrics_df)

    return confusion_metrics_df


def prepare_training_set(ssd_data, command_data, participant_ids):
    """ FILTER COMMANDS """
    run_ids = 'all'  #['R1'] #'all'
    if settings.target_type == 'direction':
        command_types = ['HDG']
    elif settings.target_type == 'geometry':
        command_types = ['HDG', 'SPD']
        command_data = command_data[command_data.preference != 'N/A']
    elif settings.target_type == 'relative_heading':
        command_types = ['HDG', 'DCT']
        command_data = command_data[command_data.hdg_rel != 'N/A']
    elif settings.target_type == 'command_type':
        command_types = ['HDG', 'SPD', 'DCT']
        settings.num_classes =  len(command_types)

    command_data = command_data[command_data.ssd_id != 'N/A']
    command_data = command_data.reset_index().set_index('ssd_id').sort_index()


    # ssd = 100
    # show_ssd(ssd_id=ssd, ssd_stack=ssd_data)
    # print(command_data.loc[ssd])

    if participant_ids[0] != 'all':
        command_data = command_data[command_data.participant_id.isin(participant_ids)]
    if run_ids is not 'all':
        command_data = command_data[command_data.run_id.isin(run_ids)]
    if command_types is not 'all':
        command_data = command_data[command_data.type.isin(command_types)]
    if settings.ssd == 'ON' or settings.ssd == 'OFF':
        command_data = command_data[command_data.SSD == settings.ssd]


    # filter ssds based on remaining actions
    actions_ids = command_data.index.unique()
    x_data = ssd_data[actions_ids, :, :, :]

    target_list = target_data_preparation.make_categorical_list(command_data, settings.target_type)

    y_data = keras.utils.to_categorical(target_list, settings.num_classes)

    return x_data, y_data


def split_data(x_data, y_data):
    train_length = int(settings.train_val_ratio * len(x_data))
    x_train = x_data[0:train_length, :, :, :]
    x_val = x_data[train_length:, :, :, :]

    y_train = y_data[0:train_length, :]
    y_val = y_data[train_length:, :]

    data_length = len(x_train) + len(x_val)
    print('Number of SSDs to network: {} ({} train / {} val)'.format(data_length, len(x_train), len(x_val)))

    return x_train, y_train, x_val, y_val


def create_model(iteration_name):
    """ CREATING THE NEURAL NETWORK """
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=settings.ssd_shape))

    # BASELINE ARCHITECTURE
    model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1)))
    convout = Activation('relu')
    model.add(convout)
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu'))

    # Flattening and FC
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    if settings.dropout_rate != 0:
        model.add(Dropout(settings.dropout_rate))
    model.add(Dense(settings.num_classes, activation='softmax'))

    # Freeze the first X layers
    if settings.freeze_layers:
        for layer in model.layers[:4]:
            layer.trainable = False

    model.summary()

    # Output model structure to disk
    if not path.exists(settings.output_dir + '/figures'):
        print('Output directory created')
        makedirs(settings.output_dir + '/figures')

    # plot_model(model, to_file='{}/figures/structure_{}.png'.format(settings.output_dir, iteration_name), show_shapes=True, show_layer_names=False)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # sgd = keras.optimizers.SGD(lr=0.001)
    # model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

    if settings.load_weights is not False:
        weights_filepath = settings.output_dir + '/weights/' + settings.load_weights + '.hdf5'
        model.load_weights(weights_filepath, by_name=False)
        print('Weights loaded from: ', weights_filepath)


    return model, convout


def create_pretrained_model():

    # Load the VGG model
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=settings.ssd_shape)

    # Freeze the layers except the last 4 layers
    if settings.freeze_layers:
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

    n = convolutions.shape[2]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12, 8))
    for i in range(convolutions.shape[2]):
        ax = fig.add_subplot(n, n, i + 1)
        ax.imshow(convolutions[:,:,i], cmap='gray')
    plt.savefig('{}/figures/filters.png'.format(settings.output_dir))
    plt.close()
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.imshow(convolutions[:,:,1], cmap='gray')
    # plt.show()


def save_training_data(informedness_list):
    filename = '{}/train_time_{}.csv'.format(settings.output_dir, settings.iteration_name)
    with open(filename, 'a') as f:
        f.write(",".join(map(str, informedness_list)))
        f.write("\n")

def show_ssd(ssd_id, ssd_stack):
    ssd = ssd_stack[ssd_id, :, :, :]
    ssd *= 255
    ssd = ssd.astype('uint8')
    ssd = Image.fromarray(ssd)
    ssd.show()


if __name__ == "__main__":
    try:
        all_data = pickle.load(open(settings.data_folder + 'all_dataframes_3.p', "rb"))
        print('Data loaded from pickle')
    except FileNotFoundError:
        print('Start importing data from image files')
        all_data = ssd_loader()

    ssd_trainer(all_data, participant_ids=['all'])

    # TODO: color channels
    # TODO: add secondary commands
    # TODO: vary over number of locked layers with transfer learning
    # bulid a numerical approximator instead of class approximator