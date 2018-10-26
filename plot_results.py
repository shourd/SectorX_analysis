import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import settings

sns.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_results(experiment_name):
    results = pd.read_csv(settings.output_dir + '/metrics_{}.csv'.format(experiment_name)).reset_index()

    participant_list = list(results.participant.unique())
    target_type_list = list(results.target_type.unique())

    max_df = pd.DataFrame()
    for participant_id in np.unique(participant_list):
        for target_type in np.unique(target_type_list):
            results_temp = results[results.participant == participant_id]
            results_temp = results_temp[results_temp.target_type == target_type]

            max_dict = {
                'participant': [participant_id],
                'target_type': [target_type],
                'ssd': ['all'],
                'mcc_max': round(results_temp.MCC.max(),2),
                'acc_max': round(results_temp.val_acc.max(),2),
                'f1_score_max': round(results_temp.val_F1_score.max(),2),
                'informedness_max': round(results_temp.val_informedness.max(),2)
            }

            max_df_temp = pd.DataFrame.from_dict(max_dict)

            if max_df.empty:
                max_df = max_df_temp
            else:
                max_df = max_df.append(max_df_temp)

    print(max_df.to_string())

    metric_list = ['mcc', 'acc', 'f1_score', 'informedness']

    fig, (ax_vars) = plt.subplots(1, 4, figsize=settings.figsize4)
    for i_metric, metric in enumerate(metric_list):
        metric = metric + '_max'
        sns.boxplot(x='target_type', y=metric, data=max_df, ax=ax_vars[i_metric])
        ax_vars[i_metric].set_ylim([0, 1])

    plt.savefig('{}/{}_max.png'.format(settings.output_dir, experiment_name), bbox_inches='tight')
    print('Boxplots saved')
    # todo: HUE SSD !


    """ SORTED PER PARTICIPANT """
    for metric in ['MCC', 'val_acc']:
        g = sns.relplot(x='epoch', y=metric, hue='target_type', col='participant', col_wrap=3, kind='line', data=results)
        # g.fig.subplots_adjust(top=.9)
        plt.savefig('{}/{}_{}.png'.format(settings.output_dir, experiment_name, metric), bbox_inches='tight')
        plt.close()
        print('{} values saved'.format(metric))

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=settings.figsize4)

    # sns.lineplot(x='epoch', y='MCC', data=results, hue='participant', ax=ax1, legend='brief')
    # # ax1.set_ylim([-0.25, 1])
    # ax1.set_ylabel('MCC')
    #
    # # box = ax1.get_position()  # get position of figure
    # # ax1.set_position([box.x0, box.y0 + box.height * 0.1,
    # #                  box.width, box.height * 0.9])
    # ax1.legend(loc='upper center', bbox_to_anchor=(2.3, 1.15), ncol=12)
    #
    # sns.lineplot(x='epoch', y='val_acc', data=results, hue='participant', ax=ax2, legend=False)
    # # ax2.set_ylim([0.4, 1])
    # ax2.set_ylabel('Accuracy')
    #
    # sns.lineplot(x='epoch', y='val_informedness', hue='participant', data=results, ax=ax3, legend=False)
    # # ax3.set_ylim([0.4, 1])
    # ax3.set_ylabel('Informedness')
    #
    # sns.lineplot(x='epoch', y='val_F1_score', hue='participant', data=results, ax=ax4, legend=False)
    # # ax4.set_ylim([0.4, 1])
    # ax4.set_ylabel('F1 score')
    # plt.savefig('{}/{}_p.png'.format(settings.output_dir, experiment_name), bbox_inches='tight')

    # """ SORTED PER TARGET TYPE """
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=settings.figsize4)
    #
    # sns.lineplot(x='epoch', y='MCC', data=results, hue='target_type', ax=ax1, legend='brief')
    # # ax1.set_ylim([-0.25, 1])
    # ax1.set_ylabel('MCC')
    #
    # # box = ax1.get_position()  # get position of figure
    # # ax1.set_position([box.x0, box.y0 + box.height * 0.1,
    # #                  box.width, box.height * 0.9])
    # ax1.legend(loc='upper center', bbox_to_anchor=(2.3, 1.15), ncol=12)
    #
    # sns.lineplot(x='epoch', y='val_acc', data=results, hue='target_type', ax=ax2, legend=False)
    # # ax2.set_ylim([0.4, 1])
    # ax2.set_ylabel('Accuracy')
    #
    # sns.lineplot(x='epoch', y='val_informedness', hue='target_type', data=results, ax=ax3, legend=False)
    # # ax3.set_ylim([0.4, 1])
    # ax3.set_ylabel('Informedness')
    #
    # sns.lineplot(x='epoch', y='val_F1_score', hue='target_type', data=results, ax=ax4, legend=False)
    # # ax4.set_ylim([0.4, 1])
    # ax4.set_ylabel('F1 score')
    # plt.savefig('{}/{}_t.png'.format(settings.output_dir, experiment_name), bbox_inches='tight')

if __name__ == '__main__':
    experiment_name = 'full_experiment_test'
    plot_results(experiment_name)
    # plot_results(settings.experiment_name)
