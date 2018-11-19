from os import makedirs

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plot_data import set_plot_settings

from config import settings

sns.set()
# matplotlib.use('macOsX')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_results(experiment_name):
    # settings
    set_plot_settings()
    sns.set_palette('Blues')
    # sns.set_context("notebook")

    target_type_order = ['type', 'direction', 'value']
    # metric_list = ['MCC', 'val_acc', 'val_F1_score', 'val_informedness']

    # start
    print('Analyzing metrics_{}.csv'.format(experiment_name))
    results = pd.read_csv(settings.output_dir + '/metrics_{}.csv'.format(experiment_name)).reset_index()
    settings.output_dir = settings.output_dir + '/' + experiment_name
    makedirs(settings.output_dir, exist_ok=True)


    """ REMOVE GEOMETRY"""
    results = results[results.target_type != 'geometry']


    """ CALCULATE RUN PERFORMANCE """
    # calculate performance metric
    performance_matrix = results.groupby(['participant', 'target_type', 'repetition']).MCC.agg('max').unstack().reset_index()
    performance_matrix['average_MCC'] = performance_matrix.loc[:, results.repetition.unique()].mean(axis=1)
    performance_matrix.to_csv('{}/{}_performance.csv'.format(settings.output_dir, experiment_name))
    # print('Performance metrics saved')

    # results_target_type = results.groupby(['participant', 'target_type','SSD']).agg('mean').reset_index()
    results_per_kfold = results.groupby(['skill_level','participant', 'target_type', 'repetition','SSD']).agg('max').reset_index()
    results_per_kfold.drop(['index', 'Unnamed: 0', 'epoch'], axis=1, inplace=True)

    results_avg_kfold = results_per_kfold.groupby(['skill_level','participant', 'target_type','SSD']).agg('mean').reset_index()
    results_avg_kfold.drop(['repetition'], axis=1, inplace=True)

    # results_average = results_repetition.groupby(['participant', 'skill_level','repetition']).agg('mean').reset_index()
    # P11 --> outlier.
    # results_average.loc[results_average.participant == 11, 'skill_level'] = 'novice'
    # results_ssd = results_repetition.groupby(['participant', 'skill_level', 'SSD', 'repetition']).agg('mean').reset_index()

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
    sns.set('paper', 'darkgrid', rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8, 'axes.titlesize': 10,
                                     'xtick.labelsize': 8,
                                     'ytick.labelsize': 8, "pgf.rcfonts": False})
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})

    # Performance per participant
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    sns.boxplot(x='participant', y='MCC', palette='Blues', data=results_per_kfold,
                linewidth=1, fliersize=2, ax=ax)
    ax.set_ylim([0, 1.1])
    ax.set_xlabel('Participant')
    plt.savefig('{}/{}_participant.pdf'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.savefig('{}/perf_participant.pgf'.format(settings.output_dir), bbox_inches='tight')
    plt.close()
    print('Plot saved')


    # performance per target type
    # g = sns.catplot(x='target_type', y='MCC', kind='box', hue='SSD', hue_order=['OFF', 'ON', 'BOTH'],
    #                 palette='muted', data=results_target_type)
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    sns.boxplot(x='target_type', y='MCC', ax=ax, linewidth=1, fliersize=2,
                    palette='Blues', data=results_avg_kfold)
    plt.ylim([0, 1.1])
    ax.set_xlabel('Abstraction level')
    plt.savefig('{}/{}_targettype.pdf'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.savefig('{}/perf_target_type.pgf'.format(settings.output_dir), bbox_inches='tight')
    plt.close()
    print('Plot saved')

    """ CONDITIONS """

    # # performance per skill level
    # fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    # sns.boxplot(x='skill_level', order=['novice', 'intermediate'], y='MCC', palette='Blues', linewidth=1, fliersize=2, data=results_avg_kfold, ax=ax)
    # plt.ylim([0, 1.1])
    # ax.set_xlabel('Skill level')
    # plt.savefig('{}/{}_skilllevel.pdf'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    # plt.savefig('{}/perf_skill_level.pgf'.format(settings.output_dir), bbox_inches='tight')
    # plt.close()
    # print('Plot saved')

    # performance per condition
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    sns.boxplot(x='SSD', order=['OFF','ON', 'BOTH'], y='MCC', hue='skill_level', hue_order=['novice', 'intermediate'],
                palette='Blues', linewidth=1, fliersize=2,
                data=results_avg_kfold, ax=ax)
    plt.ylim([0, 1.1])
    ax.legend_.set_title('Skill level')
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=2)
    plt.savefig('{}/{}_conditions.pdf'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.savefig('{}/perf_conditions.pgf'.format(settings.output_dir), bbox_inches='tight')
    plt.close()
    print('Plot saved')


    """ OVER TIME """
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article_high)
    sns.lineplot(x='epoch', y='MCC', hue='target_type', hue_order=['type', 'direction', 'value'],
                palette='muted', data=results[results.participant == 1], ax=ax, legend='brief')
    # plt.ylim([0, 1.1])
    # plt.xlim([1, 15])
    ax.set_xlabel('Epoch')
    leg_handles = ax.get_legend_handles_labels()[0]
    leg_handles = leg_handles[1:]
    legend_labels = ['Type', 'Direction', 'Value']
    plt.legend(leg_handles, legend_labels, loc='lower right', bbox_to_anchor=(1, 1), ncol=4)
    # ax.legend(leg_handles, ['Training', 'Validation'], title='Data type')

    plt.savefig('{}/{}_epochs_p1.pdf'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    plt.savefig('{}/perf_epochs_p1.pgf'.format(settings.output_dir), bbox_inches='tight')
    plt.close()
    print('Plot saved')

    # MCC per participant per target type
    g = sns.catplot(x='target_type', y='MCC', hue='SSD',  hue_order=['OFF', 'ON', 'BOTH'],col='participant',
                    col_wrap=3, kind='box', data=results_repetition,
                    order=target_type_order, height=3, aspect=1, palette='muted')
    # plt.suptitle('Only best epoch per repetition')
    plt.savefig('{}/{}_{}.png'.format(settings.output_dir, experiment_name, 'catplot'), bbox_inches='tight')
    plt.close()
    print('Plot saved')

    # MCC and Val ACC per participant over time
    for metric in ['MCC', 'val_acc']:
        g = sns.relplot(x='epoch', y=metric, hue='target_type', col='participant', col_wrap=3, kind='line', data=results,
                        height=3, aspect=1)
        # g.fig.subplots_adjust(top=.9)
        plt.savefig('{}/{}_epochs_{}.png'.format(settings.output_dir, experiment_name, metric), bbox_inches='tight')
        plt.close()
        print('{} values saved'.format(metric))


if __name__ == '__main__':
    experiment_name = 'kfold7'
    plot_results(experiment_name)
    # plot_results(settings.experiment_name)
