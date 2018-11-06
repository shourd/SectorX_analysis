import keras
import numpy as np
import pandas as pd

from command_predictor import load_model, load_weights, load_test_data
from config import settings
from strategy_trainer import prepare_training_set


def main():
    # settings.participants = np.arange(1,13,1)  # [1 .. 12]
    settings.ssd_conditions = ['BOTH']
    target_types = ['type', 'direction', 'value']

    commands, ssd_stack = load_test_data()
    participant_ids = np.arange(1,13,1)
    validation_scores_df = pd.DataFrame()

    for participant_id in participant_ids:
        print('Participant', participant_id)
        participant_scores_dict = {'type': 0, 'direction': 0, 'value': 0}
        for target_type in target_types:
            settings.target_type = target_type
            class_names = determine_class_names(target_type)

            x_data, y_data = prepare_training_set(ssd_stack, commands, [participant_id])

            model = load_model('test_3classes')
            model = load_weights(model, '{}_all_general_model.hdf5'.format(target_type))

            prediction = model.predict(x_data)
            # print(prediction)

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy', 'matthews_correlation'])

            test_score = model.test_on_batch(x_data, y_data, sample_weight=None)
            participant_scores_dict[target_type] = [test_score[2]]
            print('--------------------------------------------------')

        validation_scores_df = validation_scores_df.append(pd.DataFrame.from_dict(participant_scores_dict))


        # print(model.metrics_names[2], test_score[2])

    validation_scores_df.index = participant_ids
    validation_scores_df.to_csv('{}/test_scores.csv'.format(settings.output_dir))
    print(validation_scores_df.to_string())

        # prediction_dict[target_type] = class_names[int(np.argmax(prediction, axis=1))]
        # prediction_certainty_dict[target_type+'_certainty'] = round(float(np.max(prediction, axis=1)), 2)


def determine_class_names(target_type):
    if target_type == 'type':
        class_names = ['HDG', 'SPD', 'DCT']
    elif target_type == 'direction':
        class_names = ['Left', 'Right']
    elif target_type == 'value':
        class_names = ['0-10 deg', '10-45 deg', '> 45 deg']

    return class_names


if __name__ == '__main__':
    main()

