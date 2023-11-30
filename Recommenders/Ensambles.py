from Recommenders.BaseRecommender import BaseRecommender

import numpy as np


class LinearCombination(BaseRecommender):
    """Linear Combination Ensamble Recommender"""

    RECOMMENDER_NAME = "Linear_Combination_Ensamble_Recommender_Class"

    def __init__(self, URM_train, recommenders_list= None, hyperparameters_dicts_list= None, weights_list= None, verbose = True):
        super(LinearCombination, self).__init__(URM_train, verbose = verbose)

        self.n_recommenders = len(recommenders_list)
        self.recommenders_list = recommenders_list # list of initialized recommenders
        self.hyperparameters_dicts_list = hyperparameters_dicts_list
        if weights_list == None:
            self.weights_list = [1/self.n_recommenders] * self.n_recommenders # uniform weights if not specified
        else: self.weights_list = weights_list

    def fit(self):
        '''
            Fit each of the Recommender objects in the ensamble by calling fit() method for each of them.
        '''
        for recommender in range(len(self.recommenders_list)):
            hyperparams = self.hyperparameters_dicts_list[recommender]
            recommender_object = self.recommenders_list[recommender]
            recommender_object.fit(**hyperparams)
            print("Successfully fitted Recommender", recommender+1, ":", recommender_object.RECOMMENDER_NAME)
    
    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):
        '''
            Compute the recommendations by weighted-averaging over the scores computed by each of the 
            Recommenders in the Ensamble.
        '''

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        cutoff = min(cutoff, self.URM_train.shape[1] - 1)

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array

        # For each Recommender object we compute the scores and store them in scores_batch_array
        scores_batch_array = np.array([recommender_object._compute_item_score(user_id_array, items_to_compute=items_to_compute) 
                                       for recommender_object in self.recommenders_list])

        # Now compute the ensamble scores by calculating a weighted average over the scores of each Recommender
        scores_batch = np.average(scores_batch_array, axis= 0, weights= self.weights_list)

        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])


        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_custom_items_flag:
            scores_batch = self._remove_custom_items_on_scores(scores_batch)

        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index
        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = np.argpartition(-scores_batch, cutoff-1, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()



        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]


        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list



