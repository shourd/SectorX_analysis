import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# import statsmodels.formula.api as smf
# import statsmodels.api as sm



from config import settings
sns.set()

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.05, point['y']+.01, str(point['val']), color='0.2', fontsize='12')

def compare_performance_consistency():

    consistency_df = pd.read_csv(settings.output_dir + '/consistency_metrics_normalized.csv').set_index('participant')
    performance_df = pd.read_csv(settings.output_dir + '/performance_metrics.csv').set_index('participant')

    consistency_df.drop('Unnamed: 0', axis=1, inplace=True)
    performance_df.drop('Unnamed: 0', axis=1, inplace=True)

    """ PLOTTING """
    performance_df_melt = performance_df.reset_index().melt(id_vars='participant')
    g = sns.catplot(x='participant', y='value', hue='variable', kind='bar', palette='muted',
                    data=performance_df_melt)
    plt.title('Best obtained MCC values per abstraction level')
    plt.savefig('{}/{}/{}.png'.format(settings.output_dir, settings.experiment_name, 'performance_scores'), bbox_inches='tight')
    plt.close()

    combined_df = pd.concat([consistency_df, performance_df], axis=1, sort=False, join='outer')
    combined_df.reset_index(inplace=True)
    combined_df.participant = ['P{}'.format(int(p)) for p in combined_df.participant]

    # OUTLIERS:
    # todo: true?
    combined_df.loc[10-1, 'geometry'] = np.nan

    print(combined_df.to_string())

    # ols_report = smf.ols('average_MCC ~ final_consistency', data=combined_df).fit()
    # print(ols_report.summary())

    target_type_order = ['geometry', 'type', 'value']
    for target_type in target_type_order:
        consistency_type = target_type + '_consistency'
        g = sns.lmplot(x=consistency_type, y=target_type, data=combined_df)

        g.set_xlabels(target_type + ' consistency')
        g.set_ylabels(target_type + ' performance')
        g.set_titles(target_type)
        label_point(combined_df[consistency_type], combined_df[target_type], combined_df.participant, plt.gca())
        # plt.savefig('{}/{}/{}.png'.format(settings.output_dir, settings.experiment_name, 'regression_{}'.format(target_type)),
        #             bbox_inches='tight')
        plt.close()


    g = sns.lmplot(x='final_consistency', y='average_MCC', data=combined_df)
    plt.title('Performance - Consistency relation')
    label_point(combined_df.final_consistency, combined_df.average_MCC, combined_df.participant, plt.gca())
    plt.savefig('{}/{}/{}.png'.format(settings.output_dir, settings.experiment_name, 'performance_consistency'),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    compare_performance_consistency()
