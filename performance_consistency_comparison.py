import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from config import settings

warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))
sns.set()


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.05, point['y']+.01, str(point['val']), color='0.2', fontsize='10')


def compare_performance_consistency():

    consistency_df = pd.read_csv(settings.output_dir + '/consistency/consistency_metrics_normalized.csv').set_index('participant')
    consistency_df.drop('Unnamed: 0', axis=1, inplace=True)
    # from validation data:
    performance_df = pd.read_csv(settings.output_dir + '/{}/performance.csv'.format(settings.experiment_name)).set_index('participant')
    # from test data:
    performance_df = pd.read_csv('{}/test_scores/test_scores_{}.csv'.format(settings.output_dir, 'auto')).set_index('participant')

    """ MERGING CONSISTENCY AND PERFORMANCE """
    combined_df = pd.concat([consistency_df, performance_df], axis=1, sort=False, join='outer')
    combined_df.reset_index(inplace=True)
    combined_df.participant = ['P{}'.format(int(p)) for p in combined_df.participant]
    combined_df.drop('Unnamed: 0', axis=1, inplace=True)

    print(combined_df.to_string())
    combined_df.to_csv('{}/test_scores/cons_perf_dataframe.csv'.format(settings.output_dir))

    ols_report = smf.ols('mcc_mean ~ mean_consistency', data=combined_df).fit()
    # print(ols_report.summary())
    print('R2: ', round(ols_report.rsquared, 3))

    # target_type_order = ['direction', 'type', 'value']
    # for target_type in target_type_order:
    #     consistency_type = target_type + '_consistency'
    #     # target_type = 'direction' if target_type == 'geometry' else target_type
    #     g = sns.lmplot(x=consistency_type, y=target_type, data=combined_df)
    #
    #     g.set_xlabels(target_type + ' consistency')
    #     g.set_ylabels(target_type + ' performance')
    #     g.set_titles(target_type)
    #     label_point(combined_df[consistency_type], combined_df[target_type], combined_df.participant, plt.gca())
    #     # if target_type == 'geometry':
    #         # plt.ylim([0.6, 1.0])
    #         # plt.xlim([-1.5, 1])
    #     if target_type == 'type':
    #         plt.ylim([0.5, 1.0])
    #     if target_type == 'value':
    #         plt.ylim([0.5, 1])
    #     plt.savefig('{}/{}/{}.pdf'.format(settings.output_dir, settings.experiment_name, 'performance_consistency_{}'.format(target_type)))
    #     plt.close()

    print('Consistency')
    print(list(combined_df.mean_consistency.round(3)))
    print('MCC')
    print(list(combined_df.mcc_mean.round(3)))


    """ PLOTTING """
    # settings
    sns.set()  # reset all settings.
    sns.set('paper', 'whitegrid',
            rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
                'xtick.labelsize': 8, 'ytick.labelsize': 8},
            font='Times New Roman')

    # performance plot
    performance_df_melt = performance_df.reset_index().melt(id_vars='participant')
    g = sns.catplot(x='participant', y='value', hue='variable', kind='bar', palette='muted',
                    data=performance_df_melt, aspect=2)
    plt.title('Average MCC per abstraction level')
    plt.savefig('{}/{}/{}.pdf'.format(settings.output_dir, settings.experiment_name, 'performance_scores'), bbox_inches='tight')
    plt.close()

    # Comparison plot
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article_high)
    sns.regplot(x='mean_consistency', y='mcc_mean', robust=True, ci=None, data=combined_df, ax=ax,
                line_kws={'linewidth': 2})
    # plt.title('Performance - Consistency relation')
    label_point(combined_df.mean_consistency, combined_df.mcc_mean, combined_df.participant, plt.gca())
    # ax.set_ylim([0.5, 1])
    ax.set_xlabel('Participant Consistency')
    ax.set_ylabel('Mean MCC')
    ax.set_ylim([0.4, 0.8])
    plt.savefig('{}/{}/{}.pdf'.format(settings.output_dir, settings.experiment_name, 'performance_consistency'),
                bbox_inches='tight')
    print('Saved: {}/{}/{}.pdf'.format(settings.output_dir, settings.experiment_name, 'performance_consistency'))
    plt.close()


def delta_general_individual():
    comparison_df = pd.read_csv(settings.output_dir + '/test_scores/test_scores_summary.csv').set_index(
        'participant')

    comparison_df['delta_mcc'] = comparison_df.individual_mcc - comparison_df.general_mcc
    comparison_df['delta_acc'] = comparison_df.individual_acc - comparison_df.general_acc


if __name__ == '__main__':
    compare_performance_consistency()
    delta_general_individual()
