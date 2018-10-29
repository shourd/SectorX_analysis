import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import settings

sns.set()
consistency_df = pd.read_csv(settings.output_dir + '/consistency_metrics.csv').set_index('participant')
consistency_df.index = consistency_df.index.astype('str')
performance_df = pd.read_csv(settings.output_dir + '/performance_metrics.csv').set_index('participant')

consistency_df.drop('Unnamed: 0', axis=1, inplace=True)
performance_df.drop('Unnamed: 0', axis=1, inplace=True)

combined_df = pd.concat([consistency_df, performance_df], axis=1, sort=False, join='outer')
# test2 = consistency_df.join(performance_df, rsuffix='_perf', lsuffix='_cons', how='outer', sort=False).reset_index()

combined_df = combined_df.iloc[:-2, :]

print(combined_df.to_string())

g = sns.lmplot(x='final_consistency', y='average_performance', data=combined_df)

target_type_order = ['type', 'direction', 'value']
for target_type in target_type_order:
    g = sns.lmplot(x=target_type + '_consistency', y=target_type, data=combined_df)

plt.show()

