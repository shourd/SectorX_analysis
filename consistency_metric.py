import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import settings

warnings.simplefilter(action='ignore', category=UserWarning)


def calc_consistency_metric():
    all_data = pickle.load(open(settings.data_folder + settings.input_file, "rb"))
    commands_df = all_data['commands'].reset_index()
    commands_df = commands_df[commands_df.ssd_id != 'N/A']

    commands_temp = commands_df[commands_df.hdg_rel != 'N/A']
    total_unique = len(commands_temp.hdg_rel.unique())

    # consistency_list = []
    consistency_df = pd.DataFrame()
    for participant in range(1,13):
        commands_p = commands_df[commands_df.participant_id == participant]

        """ 1: Type of command """
        spd_commands = len(commands_p[commands_p.type == 'SPD'])
        hdg_commands = len(commands_p[commands_p.type == 'HDG'])
        type_consistency = normalize_consistency(spd_commands, hdg_commands)

        """ 2: Direction """
        commands_temp = commands_p[commands_p.type == 'HDG']
        left_commands = len(commands_temp[commands_temp.direction == 'left'])
        right_commands = len(commands_temp[commands_temp.direction == 'right'])
        increase_commands = len(commands_p[commands_p.direction == 'increase'])
        decrease_commands = len(commands_p[commands_p.direction == 'decrease'])

        direction_consistency = normalize_consistency(left_commands, right_commands)

        """ 2a: Geometry """
        # commands_temp = commands_p[commands_p.preference != 'N/A']
        # commands_temp = commands_temp[commands_temp.type.isin(['HDG', 'SPD'])]
        # behind_commands = len(commands_temp[commands_temp.preference == 'behind'])
        # infront_commands = len(commands_temp[commands_temp.preference == 'infront'])
        #
        # geometry_consistency = normalize_consistency(behind_commands, infront_commands)
        # geometry_consistency = threshold_consistency(behind_commands, infront_commands)


        """ 3. Value """
        commands_temp = commands_p[commands_p.hdg_rel != 'N/A']
        unique = len(commands_temp.hdg_rel.unique())

        value_consistency = total_unique / unique

        """ WEIGHTED TOTAL """
        # mean_consistency = np.mean([type_consistency, value_consistency, direction_consistency])
        # consistency_list.append(mean_consistency)

        dict_temp = {
            'participant': [participant],
            # 'geometry_consistency': [geometry_consistency],
            'type_consistency': [type_consistency],
            'direction_consistency': [direction_consistency],
            'value_consistency': [value_consistency],
            # 'mean_consistency': [mean_consistency]
        }

        run_df = pd.DataFrame.from_dict(dict_temp, orient='columns')

        if consistency_df.empty:
            consistency_df = run_df
        else:
            consistency_df = consistency_df.append(run_df)

    consistency_df_normalized = consistency_df.apply(stats.zscore)
    consistency_df_normalized.participant = np.arange(1,13,1)
    consistency_df_normalized['mean_consistency'] = consistency_df_normalized.loc[:, ['type_consistency', 'direction_consistency', 'value_consistency']].mean(axis=1)

    # consistency_df.to_csv(settings.output_dir + '/consistency/consistency_metrics.csv')
    consistency_df_normalized.to_csv(settings.output_dir + '/consistency/consistency_metrics_normalized.csv')

    # add skill level
    # skill_level_list = ['novice' if (p in np.arange(1,7,1)) or (p == 11) else 'intermediate' for p in consistency_df.participant]
    skill_level_list = ['novice' if (p in np.arange(1, 7, 1)) else 'intermediate' for p in consistency_df.participant]
    # skill_level_list[10] = 'outlier'
    consistency_df_normalized['skill_level'] = skill_level_list

    print(consistency_df_normalized.to_string())


    """ PLOTTING """
    consistency_df_normalized.rename(columns={'type_consistency': 'Type', 'direction_consistency': 'Direction',
                                              'value_consistency': 'Value', 'mean_consistency': 'Mean'}, inplace=True)
    consistency_df_melt = consistency_df_normalized.melt(id_vars=['participant', 'skill_level'])
    consistency_df_melt.rename(columns={'variable': 'target_type'}, inplace=True)

    sns.set('paper', 'ticks', rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
                                    'axes.titlesize': 10,
                                     'xtick.labelsize': 8,
                                     'ytick.labelsize': 8, "pgf.rcfonts": False})
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
    boxplot_linewidth = 0.5

    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    sns.barplot(data=consistency_df_melt, x='participant', y='value', hue='target_type', palette='Blues', ax=ax)
    sns.despine()
    ax.set_xlabel('Participant')
    ax.set_ylabel('Consistency (normalized)')
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=4, title='Abstraction Level')
    plt.savefig('{}/consistency/{}.pdf'.format(settings.output_dir, 'consistency_scores'), bbox_inches='tight')
    if settings.save_as_pgf:
        plt.savefig('{}/consistency/{}.pgf'.format(settings.output_dir, 'consistency_scores'), bbox_inches='tight')
    plt.close()


    # CONSISTENCY PER SKILL LEVEL
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    g = sns.boxplot(data=consistency_df_melt, x='target_type', y='value', hue='skill_level',
                    hue_order=['novice','intermediate'],
                    palette='Blues', linewidth=boxplot_linewidth, fliersize=2, ax=ax)
    sns.despine()
    ax.set_xlabel('Abstraction Level')
    ax.set_ylabel('Consistency (normalized)')
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=2, title='Skill Level')
    plt.savefig('{}/consistency/{}.pdf'.format(settings.output_dir, 'consistency_scores_skill'),
                bbox_inches='tight')
    if settings.save_as_pgf:
        plt.savefig('{}/consistency/{}.pgf'.format(settings.output_dir, 'consistency_scores_skill'),
                    bbox_inches='tight')
    plt.close()

def normalize_consistency(x, y):
    total = x + y
    return 2 * (np.max([x/total, y/total]) - 0.5)

# def threshold_consistency(x, y):
#     total = x + y
#     return 2 * (np.min([x/total, y/total]))


if __name__ == '__main__':
    calc_consistency_metric()
