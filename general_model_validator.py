import cross_validator
import pandas as pd
from config import settings

def main():

    all_results_df = pd.DataFrame()

    number_of_models = 5
    for i in range(number_of_models):
        weights_folder = 'general{}'.format(i+1)
        model_weights = 'all'
        model_results_df = cross_validator.main(model_weights, weights_folder=weights_folder)
        all_results_df = all_results_df.append(model_results_df)

    print(all_results_df)
    all_results_df.to_csv('{}/test_scores/test_scores_generalmodels.csv'.format(settings.output_dir))


if __name__ == '__main__':
    main()