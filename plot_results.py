from os import makedirs

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import settings

sns.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_results(experiment_name):
    # settings
    target_type_order = ['geometry', 'type', 'direction', 'value']
    metric_list = ['MCC', 'val_acc', 'val_F1_score', 'val_informedness']
    sns.set_context("notebook")

    # start
    print('Analyzing metrics_{}.csv'.format(experiment_name))
    results = pd.read_csv(settings.output_dir + '/metrics_{}.csv'.format(experiment_name)).reset_index()

    """ CALCULATE RUN PERFORMANCE """
    # calculate performance metric
    # results['performance'] = (results.MCC + results.val_F1_score + results.val_acc + results.val_informedness)/4
    performance_matrix = sort_dataframe(results.groupby(['participant', 'target_type']).MCC.agg('max').unstack().reset_index())
    performance_matrix['average_MCC'] = performance_matrix.loc[:, target_type_order].mean(axis=1)
    performance_matrix.to_csv('{}/performance_metrics.csv'.format(settings.output_dir))
    # print('Performance metrics saved')

    results_target_type = results.groupby(['participant', 'target_type']).agg('max').reset_index()
    results_repetition = sort_dataframe(results.groupby(['participant', 'target_type', 'repetition']).agg('max').reset_index())

    mcc_mean = round(results_target_type.MCC.mean(), 2)
    mcc_mean_all_reps = round(results_repetition.MCC.mean(), 2)
    acc_mean = round(results_repetition.val_acc.mean(), 2)

    """ PRINT RUN PERFORMANCE """

    # PER TARGET TYPE
    print('-----------------------')
    print('Mean MCC per type: ', end='')
    for target_type in target_type_order:
        mean = round(results_target_type[results_target_type.target_type == target_type].MCC.mean(), 2)
        print('[{}: {}]'.format(target_type, mean), end=' ')

    print('\nAverage best MCC: {} ({}). Average accuracy: {}'.format(mcc_mean, mcc_mean_all_reps, acc_mean))
    print('-----------------------')

    """ PLOT RUN PERFORMANCE """
    settings.output_dir = settings.output_dir + '/' + experiment_name
    makedirs(settings.output_dir, exist_ok=True)

    # Performance per participant
    g = sns.catplot(x='participant', y='MCC', kind='box', palette='muted', data=results_target_type)
    plt.ylim([0, 1.1])
    plt.savefig('{}/{}_participant.png'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.close()

    # performance per skill level
    g = sns.catplot(x='skill_level', y='MCC', kind='box', hue='SSD', palette='muted', data=results_target_type)
    plt.ylim([0, 1.1])
    plt.savefig('{}/{}_skilllevel.png'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.close()

    # performance per target type
    g = sns.catplot(x='target_type', y='MCC', kind='box', hue='SSD', palette='muted', data=results_target_type)
    plt.ylim([0, 1.1])
    plt.savefig('{}/{}_targettype.png'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.close()

    # # performance per target type per validation metric
    # fig, (ax_vars) = plt.subplots(1, 4, figsize=settings.figsize4)
    # for i_metric, metric in enumerate(metric_list):
    #     sns.boxplot(x='target_type', y=metric, data=results_target_type, hue='SSD', order=target_type_order, ax=ax_vars[i_metric])
    #     ax_vars[i_metric].set_ylim([-0.1, 1.1])
    #     # hue='skill_level'
    #
    # plt.savefig('{}/{}_allmetrics.png'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    # plt.close()
    # print('Boxplots saved')

    # MCC per participant per target type
    g = sns.catplot(x='target_type', y='MCC', hue='SSD', col='participant', col_wrap=3, kind='box', data=results_repetition,
                    order=target_type_order, height=3, aspect=1, palette='muted')
    plt.suptitle('Only best epoch per repetition')
    plt.savefig('{}/{}_{}.png'.format(settings.output_dir, experiment_name, 'catplot'), bbox_inches='tight')
    plt.close()
    print('Aggregated boxplots saved')


    # MCC and Val ACC per participant over time
    for metric in ['MCC']:
        g = sns.relplot(x='epoch', y=metric, hue='target_type', col='participant', col_wrap=3, kind='line', data=results,
                        height=3, aspect=1)
        # g.fig.subplots_adjust(top=.9)
        plt.savefig('{}/{}_epochs_{}.png'.format(settings.output_dir, experiment_name, metric), bbox_inches='tight')
        plt.close()
        print('{} values saved'.format(metric))


def sort_dataframe(dataframe):
    # convert PX to X
    p_list = []
    for p in list(dataframe.participant):
        if p != 'all':
            p = int(p[1:])
        else:
            p = 999999  # 'all' -> 13
        p_list.append(p)
    dataframe.participant = p_list
    dataframe.sort_values(by=['participant'], inplace=True)
    return dataframe.replace(999999, 'all')


if __name__ == '__main__':
    experiment_name = 'input_nonoise'
    plot_results(experiment_name)
    # plot_results(settings.experiment_name)
