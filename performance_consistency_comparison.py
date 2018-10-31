import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import settings
sns.set()

target_type_order = ['geometry', 'type', 'value']

consistency_df = pd.read_csv(settings.output_dir + '/consistency_metrics_normalized.csv').set_index('participant')
# consistency_df = pd.read_csv(settings.output_dir + '/consistency_metrics.csv').set_index('participant')
consistency_df.index = consistency_df.index.astype('int')
performance_df = pd.read_csv(settings.output_dir + '/performance_metrics.csv').set_index('participant')

consistency_df.drop('Unnamed: 0', axis=1, inplace=True)
performance_df.drop('Unnamed: 0', axis=1, inplace=True)

""" PLOTTING """

# g = sns.catplot(x='participant', y='average_performance', kind='bar', palette='muted',
#                 data=performance_df.reset_index())
# plt.title('Average')

fig, (ax_vars) = plt.subplots(1, 4, figsize=settings.figsize4)
for i_target_type, target_type in enumerate(target_type_order):
    sns.catplot(x='participant', y=target_type, kind='bar', palette='muted',
                    data=performance_df.reset_index(), ax=ax_vars[i_target_type])
    ax_vars[i_target_type].set_title(target_type)
    plt.close()

plt.savefig('{}/{}.png'.format(settings.output_dir, 'performance'), bbox_inches='tight')
plt.close(fig)

combined_df = pd.concat([consistency_df, performance_df], axis=1, sort=False, join='outer')
# combined_df = combined_df.iloc[:-2, :]
print(combined_df.to_string())

target_type_order = ['geometry', 'type', 'value']
# target_type_order = ['value']
for target_type in target_type_order:
    """ OUTLIERS """
    g = sns.lmplot(x=target_type + '_consistency', y=target_type, data=combined_df)
    g.set_xlabels(target_type + ' consistency')
    g.set_ylabels(target_type + ' performance')

g = sns.lmplot(x='final_consistency', y='average_MCC', data=combined_df)

plt.show()