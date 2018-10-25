from config import settings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools
sns.set()

def plot_results(experiment_name):

    results = pd.read_csv(settings.output_dir + '/metrics_{}.csv'.format(experiment_name)).reset_index()

    """ SPLIT ITERATION NAME : CAN BE REMOVED IN NEXT RUN """
    participant_list = []
    target_type_list = []
    for iteration in list(results.iteration_name):
        try:
            target_type, p_no, _, _ = iteration.split('_')
        except ValueError:
            part1, part2, p_no, _, _ = iteration.split('_')
            target_type = part1 + '_' + part2

        participant_list.append(p_no)
        target_type_list.append(target_type)

    # results.iteration_name = participant_list
    results['participant'] = participant_list
    results['target_type'] = target_type_list



    # print('Max MCC: ',round(results.MCC.max(),2))
    # print('Max val_acc: ',round(results.val_acc.max(),2))
    # print('Max F1 score: ',round(results.val_F1_score.max(),2))
    # print('Max informedness: ',round(results.val_informedness.max(),2))

    # results.rename(columns={'iteration_name': 'Iteration name'}, inplace=True)

    # palette = itertools.cycle(sns.color_palette())

    """ SORTED PER PARTICIPANT """
    g = sns.FacetGrid(results, col="participant", col_wrap=3, palette='GnBu_d')
    g.map(sns.lineplot, "epoch", "MCC", "target_type")
    plt.suptitle('Matthews Correlation Coefficient')
    g.fig.subplots_adjust(top=.9)
    plt.savefig('{}/{}_MCC.png'.format(settings.output_dir, experiment_name), bbox_inches='tight')

    g = sns.FacetGrid(results, col="participant", col_wrap=3, palette='GnBu_d')
    g.map(sns.lineplot, "epoch", "val_acc", "target_type")
    plt.suptitle('Validation Accuracy')
    g.fig.subplots_adjust(top=.9)
    plt.savefig('{}/{}_ACC.png'.format(settings.output_dir, experiment_name), bbox_inches='tight')

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


    """ FACETPLOT """



if __name__ == '__main__':
    experiment_name = 'full_experiment'
    plot_results(experiment_name)
    # plot_results(settings.experiment_name)
