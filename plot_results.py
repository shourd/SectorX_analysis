import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import settings

sns.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_results(experiment_name):
    print('Plotting metrics_{}.csv'.format(experiment_name))
    target_type_order = ['geometry', 'type', 'direction', 'value']
    sns.set_context("notebook")
    results = pd.read_csv(settings.output_dir + '/metrics_{}.csv'.format(experiment_name)).reset_index()

    """ CALCULATE RUN PERFORMANCE """

    results_target_type = results.groupby(['participant', 'target_type']).agg('max').reset_index()
    results_repetition = results.groupby(['participant', 'target_type', 'repetition']).agg('max').reset_index()

    mcc_mean = round(results_target_type.MCC.mean(), 2)
    mcc_mean_all_reps = round(results_repetition.MCC.mean(), 2)
    acc_mean = round(results_repetition.val_acc.mean(), 2)

    print('Average best MCC: {} ({}). Average accuracy: {}'.format(mcc_mean, mcc_mean_all_reps, acc_mean))

    # convert PX to X
    p_list = []
    for p in list(results_repetition.participant):
        if p != 'all':
            p = int(p[1:])
        else:
            p = 13  # 'all' -> 13
        p_list.append(p)
    results_repetition.participant = p_list
    results_repetition.sort_values(by=['participant'], inplace=True)

    metric_list = ['MCC', 'val_acc', 'val_F1_score', 'val_informedness']

    fig, (ax_vars) = plt.subplots(1, 4, figsize=settings.figsize4)
    for i_metric, metric in enumerate(metric_list):
        sns.boxplot(x='target_type', y=metric, data=results_target_type, hue='SSD', order=target_type_order, ax=ax_vars[i_metric])
        ax_vars[i_metric].set_ylim([0, 1])
        # hue='skill_level'

    plt.savefig('{}/{}_comb.png'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.close()
    print('Boxplots saved')

    # this boxplot plots all metrics at all epochs.
    g = sns.catplot(x='target_type', y='MCC', hue='skill_level', col='participant', col_wrap=3, kind='box', data=results,
                    order=target_type_order, height=3, aspect=1)
    plt.suptitle('All epochs')
    plt.savefig('{}/{}_{}.png'.format(settings.output_dir, experiment_name, 'catplot'), bbox_inches='tight')
    plt.close()

    # this boxplot plots the best metrics per epoch.
    g = sns.catplot(x='target_type', y='MCC', hue='SSD', col='participant', col_wrap=3, kind='box', data=results_repetition,
                    order=target_type_order, height=3, aspect=1)
    plt.suptitle('Only best epoch per repetition')
    plt.savefig('{}/{}_{}.png'.format(settings.output_dir, experiment_name, 'catplot2'), bbox_inches='tight')
    plt.close()
    print('Aggregated boxplots saved')


    """ SORTED PER PARTICIPANT """
    for metric in ['MCC', 'val_acc']:
        g = sns.relplot(x='epoch', y=metric, hue='target_type', col='participant', col_wrap=3, kind='line', data=results,
                        height=3, aspect=1)
        # g.fig.subplots_adjust(top=.9)
        plt.savefig('{}/{}_{}.png'.format(settings.output_dir, experiment_name, metric), bbox_inches='tight')
        plt.close()
        print('{} values saved'.format(metric))



if __name__ == '__main__':
    experiment_name = 'baseline_dropout4'
    plot_results(experiment_name)
    # plot_results(settings.experiment_name)
