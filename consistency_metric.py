from config import settings
import pickle
import numpy as np
from scipy import stats
import pandas as pd

def calc_consistency_metric():
    all_data = pickle.load(open(settings.data_folder + settings.input_file, "rb"))
    commands_df = all_data['commands'].reset_index()

    commands_temp = commands_df[commands_df.hdg_rel != 'N/A']
    total_variance = commands_temp.hdg_rel.var(axis=0, skipna=True)
    total_unique = len(commands_temp.hdg_rel.unique())

    consistency_list = []
    consistency_df = pd.DataFrame()
    for participant in settings.columns:

        commands_p = commands_df[commands_df.participant_id == participant]
        if commands_p.empty:
            continue

        """ 1: Type of command """
        spd_commands = len(commands_p[commands_p.type == 'SPD'])
        hdg_commands = len(commands_p[commands_p.type == 'HDG'])
        type_consistency = round(normalize_consistency(spd_commands, hdg_commands), 2)

        """ 2: Direction """
        commands_temp = commands_p[commands_p.type == 'HDG']
        left_commands = len(commands_temp[commands_temp.direction == 'left'])
        right_commands = len(commands_temp[commands_temp.direction == 'right'])

        increase_commands = len(commands_p[commands_p.direction == 'increase'])
        decrease_commands = len(commands_p[commands_p.direction == 'decrease'])

        direction_hdg_consistency = normalize_consistency(left_commands, right_commands)
        direction_spd_consistency = normalize_consistency(increase_commands, decrease_commands)

        direction_consistency = round(direction_hdg_consistency, 2)
            # round((direction_hdg_consistency + direction_spd_consistency) / 2, 2)

        """ 3. Value """
        commands_temp = commands_p[commands_p.hdg_rel != 'N/A']
        variance = commands_temp.hdg_rel.var(axis=0)
        unique = len(commands_temp.hdg_rel.unique())

        variance_fraction = total_variance / variance
        unique_fraction = total_unique / unique

        w1 = 0.5
        w2 = 0.5

        value_consistency = round((w1 * variance_fraction + w2 * unique_fraction)/3, 2)

        """ WEIGHTED TOTAL """
        total_consistency = round((type_consistency + direction_consistency + value_consistency) / 3, 2)
        consistency_list.append(total_consistency)

        dict = {
            'Participant': [participant],
            'Type': [type_consistency],
            'Direction': [direction_consistency],
            'Value': [value_consistency],
            'Value (variance)': [variance_fraction],
            'Value (unique)': [unique_fraction],
            'Final': [total_consistency]
        }

        run_df = pd.DataFrame.from_dict(dict, orient='columns')

        if consistency_df.empty:
            consistency_df = run_df
        else:
            consistency_df = consistency_df.append(run_df)

        print('--------------------------')
        print('Participant: {}'.format(participant))
        print('Type: ', type_consistency)
        print('Direction: ', direction_consistency)
        print('Value: ', value_consistency)
        print('Final consistency: ', total_consistency)


    consistency_array = np.array(consistency_list)
    consistency_zscores = np.round(stats.zscore(consistency_array), decimals=2)
    print('--------------------------')
    print('Z-scores: ', consistency_zscores)
    print('Average consistency: ', round(np.array(consistency_list).mean(), 2))

    consistency_df.to_csv(settings.output_dir + '/consistency_metrics.csv')

def normalize_consistency(x, y):
    total = x + y
    return 2 * (np.max([x/total, y/total]) - 0.5)


if __name__ == '__main__':
    calc_consistency_metric()