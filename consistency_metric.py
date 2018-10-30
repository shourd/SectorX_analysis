from config import settings
import pickle
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns

def calc_consistency_metric():
    all_data = pickle.load(open(settings.data_folder + settings.input_file, "rb"))
    commands_df = all_data['commands'].reset_index()

    commands_temp = commands_df[commands_df.hdg_rel != 'N/A']
    total_unique = len(commands_temp.hdg_rel.unique())

    consistency_list = []
    consistency_df = pd.DataFrame()
    for participant in range(1,13):

        commands_p = commands_df[commands_df.participant_id == 'P{}'.format(participant)]
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

        """ 2a: Geometry """
        commands_temp = commands_p[commands_p.preference != 'N/A']
        behind_commands = len(commands_temp[commands_temp.preference == 'behind'])
        infront_commands = len(commands_temp[commands_temp.preference == 'infront'])

        geometry_consistency = normalize_consistency(behind_commands, infront_commands)

        """ 3. Value """
        commands_temp = commands_p[commands_p.hdg_rel != 'N/A']
        unique = len(commands_temp.hdg_rel.unique())

        value_consistency = total_unique / unique

        """ WEIGHTED TOTAL """
        total_consistency = round((type_consistency + direction_consistency + value_consistency) / 3, 2)
        consistency_list.append(total_consistency)

        dict = {
            'participant': [participant],
            'geometry_consistency': [geometry_consistency],
            'type_consistency': [type_consistency],
            'direction_consistency': [direction_consistency],
            'value_consistency': [value_consistency],
            'final_consistency': [total_consistency]
        }

        run_df = pd.DataFrame.from_dict(dict, orient='columns')

        if consistency_df.empty:
            consistency_df = run_df
        else:
            consistency_df = consistency_df.append(run_df)

        print('--------------------------')
        print('participant: {}'.format(participant))
        print('geometry: ', geometry_consistency)
        print('type: ', type_consistency)
        print('direction: ', direction_consistency)
        print('value: ', value_consistency)
        print('final consistency: ', total_consistency)

    consistency_df_normalized = consistency_df.apply(stats.zscore)
    consistency_df_normalized.participant = [1,2,3,4,5,6,7,8,9,10,11]
    print(consistency_df_normalized.to_string())



    # consistency_array = np.array(consistency_list)
    # consistency_zscores = np.round(stats.zscore(consistency_array), decimals=2)
    # print('--------------------------')
    # print('Z-scores: ', consistency_zscores)
    # print('Average consistency: ', round(np.array(consistency_list).mean(), 2))

    consistency_df.to_csv(settings.output_dir + '/consistency_metrics.csv')
    consistency_df_normalized.to_csv(settings.output_dir + '/consistency_metrics_normalized.csv')

    """ PLOTTING """
    g = sns.catplot(data=consistency_df, x='participant', y='final_consistency', kind='bar', palette='muted')

def normalize_consistency(x, y):
    total = x + y
    return 2 * (np.max([x/total, y/total]) - 0.5)


if __name__ == '__main__':
    calc_consistency_metric()