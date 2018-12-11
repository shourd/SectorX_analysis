from scipy import stats

import pandas as pd
import numpy as np

from config import settings


def analyze_statistics():
    test_df = pd.read_csv('{}/test_scores/test_scores_summary.csv'.format(settings.output_dir))
    perf_cons_df = pd.read_csv('{}/test_scores/cons_perf_dataframe.csv'.format(settings.output_dir))
    all_results_df = pd.read_csv('{}/{}/results_avg_kfold_all_ssd.csv'.format(settings.output_dir, 'paper_seed2'))
    # all_results_df.drop(['Unnamed: 0', 'val_F1_score', 'val_acc', axis=1, inplace=True)
    """ T TESTS SSD INFLUENCE """
    # type
    # all_results_df_type = all_results_df[all_results_df.target_type == 'type']
    # all_results_df_type.MCC


    """ T TESTS MODEL COMPARISON """
    # MCC
    results = stats.ttest_rel(test_df.general_mcc, test_df.individual_mcc)
    difference = test_df.individual_mcc - test_df.general_mcc
    # print(stats.describe(difference))
    # print(list(test_df.general_mcc))
    # print(list(test_df.individual_mcc))
    t_test_mcc = {
        't-value': round(results.statistic, 3),
        'p-value': round(results.pvalue, 4),
        'mean_diff': round(difference.mean(), 3),
        'SD': round(difference.std(), 3)
    }
    print('T-test Model Comparison - MCC')
    print('t(11) = {}, p-value = {}, mean = {}, SD = {}'.format(t_test_mcc['t-value'], t_test_mcc['p-value'],
                                                                t_test_mcc['mean_diff'], t_test_mcc['SD']))
    streep()

    # ACC
    results = stats.ttest_rel(test_df.general_acc, test_df.individual_acc)
    individual_model_acc = stats.describe(test_df.individual_acc)
    # print('individual_model_acc stats:', individual_model_acc)
    # print('general_model acc stats:', stats.describe(test_df.general_acc))
    difference = test_df.individual_acc - test_df.general_acc
    print(stats.describe(difference))
    # print(list(test_df.general_acc.round(3)))
    # print(list(test_df.individual_acc.round(3)))
    t_test_acc = {
        't-value': round(results.statistic, 3),
        'p-value': round(results.pvalue, 5),
        'mean_diff': round(difference.mean(), 3),
        'SD': round(difference.std(), 3)
    }
    print('T-test Model Comparison - Accuracy')
    print('t(11) = {}, p-value = {}, mean = {}, SD = {}'.format(t_test_acc['t-value'], t_test_acc['p-value'],
                                                                t_test_acc['mean_diff'], t_test_acc['SD']))
    streep()


    """ PEARSON'S CORRELATION COEFFICIENT """
    # print(list(perf_cons_df.mean_consistency))
    # print(list(perf_cons_df.mcc_mean))

    results = stats.pearsonr(perf_cons_df.mean_consistency, perf_cons_df.mcc_mean)
    pearson_results = {
        'r': round(results[0], 3),
        'R2': round(np.square(results[0]), 3),
        'p-value': round(results[1], 4)
    }
    print('Correlation between Consistency and Performance:')
    print('r = {}, p-value = {}'.format(pearson_results['r'], pearson_results['p-value']))


def streep():
    print('-------------------------------------------')


if __name__ == '__main__':
    analyze_statistics()
