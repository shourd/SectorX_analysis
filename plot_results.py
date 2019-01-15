from os import makedirs

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import settings
import numpy as np


def plot_results(experiment_name):
    # settings
    target_type_order = ['type', 'direction', 'value']

    # start
    print('Analyzing metrics_{}.csv'.format(experiment_name))
    results = pd.read_csv(settings.output_dir + '/metrics_{}.csv'.format(experiment_name)).reset_index()
    settings.output_dir = settings.output_dir + '/' + experiment_name
    makedirs(settings.output_dir, exist_ok=True)

    # remove geometry if existing
    results = results[results.target_type != 'geometry']

    """ CALCULATE RUN PERFORMANCE """
    # calculate performance metric
    results_temp = results[results.SSD == 'BOTH']
    performance_matrix = results_temp.groupby(['participant', 'target_type', 'repetition']).MCC.agg('max').unstack().reset_index()
    performance_matrix['mean_MCC'] = performance_matrix.loc[:, results.repetition.unique()].mean(axis=1)
    performance_matrix = performance_matrix.pivot(index='participant', columns='target_type', values='mean_MCC')
    performance_matrix['mean_MCC'] = performance_matrix.loc[:, target_type_order].mean(axis=1)
    performance_matrix.to_csv('{}/performance.csv'.format(settings.output_dir, experiment_name))
    print('Performance metrics saved')

    results_per_kfold = results.groupby(['skill_level','participant', 'target_type', 'repetition','SSD']).agg('max').reset_index()
    results_per_kfold.drop(['index', 'Unnamed: 0', 'epoch'], axis=1, inplace=True)

    results_avg_kfold_all_ssd = results_per_kfold.groupby(['skill_level', 'participant', 'target_type', 'SSD']).agg('mean').reset_index()
    results_avg_kfold_all_ssd.drop(['repetition'], axis=1, inplace=True)
    results_avg_kfold_all_ssd.to_csv('{}/results_avg_kfold_all_ssd.csv'.format(settings.output_dir, experiment_name))
    results_avg_kfold = results_avg_kfold_all_ssd[results_avg_kfold_all_ssd.SSD == 'BOTH']

    mcc_mean = round(results_per_kfold.MCC.mean(), 2)
    mcc_mean_all_reps = round(results_avg_kfold.MCC.mean(), 2)
    acc_mean = round(results_per_kfold.val_acc.mean(), 2)

    """ CALCULATE NUMBER OF SAMPLES PER TARGET PER PARTICIPANT """
    num_samples_df = results.groupby(['participant', 'target_type']).num_train_samples.agg('max').unstack().reset_index()
    num_samples_df.to_csv('{}/{}_num_samples.csv'.format(settings.output_dir, experiment_name))

    """ PRINT RUN PERFORMANCE """

    # PER TARGET TYPE
    print('-----------------------')
    print('Mean MCC per type: ', end='')
    for target_type in target_type_order:
        mean = round(results_per_kfold[results_per_kfold.target_type == target_type].MCC.mean(), 2)
        print('[{}: {}]'.format(target_type, mean), end=' ')

    print('\nAverage best MCC: {} ({}). Average accuracy: {}'.format(mcc_mean, mcc_mean_all_reps, acc_mean))
    print('-----------------------')

    """ PLOT RUN PERFORMANCE """

    sns.set()  # reset all settings.
    sns.set('paper', 'whitegrid',
            rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
                'xtick.labelsize': 8, 'ytick.labelsize': 8},
            font='Times New Roman',
            palette='Blues')
    boxplot_linewidth = 0.5

    # Performance per participant
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    sns.boxplot(x='participant', y='MCC', color=(146/255,187/255,211/255), saturation=1, data=results_per_kfold,
                linewidth=boxplot_linewidth, fliersize=2, ax=ax)
    sns.despine()
    # ax.set_ylim([0, 1.1])
    ax.set_xlabel('Participant')
    plt.savefig('{}/{}_participant.pdf'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.close()
    print('Plot saved')

    # performance per target type
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    sns.boxplot(x='target_type', y='MCC', ax=ax, linewidth=boxplot_linewidth, fliersize=2,
                palette='Blues', data=results_avg_kfold)
    sns.despine()
    plt.ylim([0, 1.1])
    ax.set_xlabel('Abstraction level')
    plt.savefig('{}/{}_targettype.pdf'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    # if settings.save_as_pgf:
    #     plt.savefig('{}/perf_target_type.pgf'.format(settings.output_dir), bbox_inches='tight')
    plt.close()
    print('Plot saved')

    """ CONDITIONS """
    # data output
    target_type = 'type'
    novice = results_avg_kfold[(results_avg_kfold.skill_level == 'novice') & (results_avg_kfold.target_type == target_type)].MCC
    intermediate = results_avg_kfold[(results_avg_kfold.skill_level == 'intermediate') & (results_avg_kfold.target_type == target_type)].MCC
    print('Novice mcc values:')
    print(list(novice))
    print('Intermediate mcc values:')
    print(list(intermediate))

    # Skill level per target type
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    sns.boxplot(x='target_type', order=['type', 'direction', 'value'], y='MCC', hue='skill_level', hue_order=['novice', 'intermediate'],
                palette='Blues', linewidth=boxplot_linewidth, fliersize=2,
                data=results_avg_kfold, ax=ax)
    sns.despine()
    plt.ylim([0, 1.1])
    leg_handles = ax.get_legend_handles_labels()[0]
    legend_labels = ['Novice', 'Intermediate']
    plt.legend(leg_handles, legend_labels, loc='lower right', bbox_to_anchor=(1, 1), ncol=2, title='Skill level')
    ax.set_xlabel('Abstraction level')
    plt.savefig('{}/perf_skill_level.pdf'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.close()
    print('Plot saved')

    # SSD per target type
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    sns.boxplot(x='target_type', order=['type', 'direction', 'value'], y='MCC', hue='SSD', hue_order=['OFF', 'ON'],
                palette='Blues', linewidth=boxplot_linewidth, fliersize=2,
                data=results_avg_kfold_all_ssd, ax=ax)
    sns.despine()
    plt.ylim([0, 1.1])
    ax.legend_.set_title('SSD')
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, title='SSD')
    ax.set_xlabel('Abstraction level')
    plt.savefig('{}/perf_conditions_ssd.pdf'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.close()
    print('Plot saved')

    """ OVER TIME """

    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article_high)
    sns.lineplot(x='epoch', y='MCC', hue='target_type', hue_order=['type', 'direction', 'value'],
                palette='muted', data=results[results.participant == 1], ax=ax, legend='brief')
    # plt.ylim([0, 1.1])
    sns.despine()
    plt.xlim([1, 25])
    ax.set_xlabel('Epoch')
    leg_handles = ax.get_legend_handles_labels()[0]
    leg_handles = leg_handles[1:]
    legend_labels = ['Type', 'Direction', 'Value']
    plt.legend(leg_handles, legend_labels, loc='lower right', bbox_to_anchor=(1, 1), ncol=4)
    # ax.legend(leg_handles, ['Training', 'Validation'], title='Data type')

    plt.savefig('{}/perf_epochs_p1.pdf'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.close()
    print('P1 Plot saved')

    # MCC per participant per target type
    results_per_kfold.rename(columns={'target_type': 'Target Type'}, inplace=True)
    sns.catplot(x='Target Type', y='MCC', hue='SSD',  hue_order=['OFF', 'ON', 'BOTH'], col='participant',
                col_wrap=3, kind='box', data=results_per_kfold,
                order=target_type_order, height=3, aspect=1, palette='Blues',
                linewidth=boxplot_linewidth)
    plt.savefig('{}/{}_{}.pdf'
                ''.format(settings.output_dir, experiment_name, 'catplot'), bbox_inches='tight')
    plt.close()
    print('Catplot saved')

    # MCC and Val ACC per participant over time
    results.rename(columns={'target_type': 'Target Type', 'val_acc': 'accuracy'}, inplace=True)
    for metric in ['MCC', 'accuracy']:
        g = sns.relplot(x='epoch', y=metric, hue='Target Type', col='participant', col_wrap=3, kind='line', data=results,
                        height=3, aspect=1, linewidth=boxplot_linewidth)
        plt.savefig('{}/{}_epochs_{}.pdf'.format(settings.output_dir, experiment_name, metric), bbox_inches='tight')
        plt.close()
        print('{} values saved'.format(metric))


if __name__ == '__main__':
    experiment_name = 'paper_seed2'  # settings.experiment_name  # 'paper_crop_64_2'
    plot_results(experiment_name)
    # plot_results(settings.experiment_name)
