from Recommenders.BaseRecommender import BaseRecommender
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample


import pandas as pd
import numpy as np
from IPython.display import display

class PredictionsWriter(object):
    ''' Object to write predictions to .csv for submission '''

    def __init__(self, recommender_object, URM, at= 10, fit_hyperparams_dict= None):
        self.recommender_object = recommender_object
        self.at = at
        self.URM = URM
        self.hyperparams_dict = fit_hyperparams_dict
        self.predictions = None
        self.evaluation_metrics_df = None



    def fit(self):
        self.recommender_object.fit(**self.hyperparams_dict)



    def evaluate_recommender(self, test_split= 0.15):
        URM_train, URM_test = split_train_in_two_percentage_global_sample(self.URM, train_percentage = (1 - test_split))
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
        result_df, _ = evaluator_test.evaluateRecommender(self.recommender_object)
        self.evaluation_metrics_df = result_df
        return result_df



    def write_predictions(self):
        prediction_df = pd.read_csv('data_target_users_test.csv', sep= ",",
                                    header=0, 
                                    dtype={0:int},
                                    engine='python')
        users = np.array(prediction_df['user_id'])
        recommendations = self.recommender_object.recommend(users)
        truncated_recommendations = [inner_list[:self.at] for inner_list in recommendations]
        prediction_df['item_list'] = truncated_recommendations
        def transform_items_to_string(item_list):
            return ' '.join(map(str, item_list))

        prediction_df['item_list'] = prediction_df['item_list'].apply(transform_items_to_string)
        print(prediction_df.head(10))
        prediction_df.to_csv('submission.csv',index=False)

        self.predictions = prediction_df



    def save_model_and_predictions(self):
        if self.predictions == None:
            self.write_predictions()
        
