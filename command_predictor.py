import pickle

import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json

from config import settings


def main():
    commands, ssd_stack = load_test_data()

    for ssd in ssd_stack:
        ssd_expanded = np.expand_dims(ssd, axis=0)

        target_types = ['type', 'direction', 'value']
        prediction_dict = {
            'type': 'None',
            'direction': 'None',
            'value': 'None'
        }
        prediction_certainty_dict = dict()
        for target_type in target_types:
            if prediction_dict['type'] == 'SPD':
                continue
            model_name = 'test_3classes'
            weights_filename = '{}_all_general_model.hdf5'.format(target_type)

            class_names = determine_class_names(target_type)

            model = load_model(model_name)
            model = load_weights(model, weights_filename)

            prediction = model.predict(ssd_expanded)
            prediction_dict[target_type] = class_names[int(np.argmax(prediction, axis=1))]
            prediction_certainty_dict[target_type+'_certainty'] = round(float(np.max(prediction, axis=1)), 2)

        print('-----------------------------------')

        print(prediction_dict)
        print(prediction_certainty_dict)
        show_ssd(ssd)


def load_model(model_name):
    """ Load JSON model from disk """
    # print("Loading {}.json".format(model_name))

    try:
        json_file = open('{}/{}.json'.format(settings.output_dir, model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        return model

    except FileNotFoundError:
        print('Model not found.')
        return


def load_test_data():
    all_data = pickle.load(open(settings.data_folder + settings.input_file, "rb"))
    commands_df = all_data['commands'].reset_index()
    commands_df = commands_df[commands_df.ssd_id != 'N/A']
    ssd_stack = all_data['ssd_images']

    return commands_df, ssd_stack


def determine_class_names(target_type):
    if target_type == 'type':
        class_names = ['HDG', 'SPD', 'DCT']
    elif target_type == 'direction':
        class_names = ['Left', 'Right']
    elif target_type == 'value':
        class_names = ['0-10 deg', '10-45 deg', '> 45 deg']

    return class_names


def show_ssd(ssd_array):
    # ssd_array *= 255
    # ssd_array = ssd_array.astype('uint8')
    # ssd = Image.fromarray(ssd_array)
    plt.imshow(np.asarray(ssd_array))
    plt.show()

if __name__ == '__main__':
    main()

