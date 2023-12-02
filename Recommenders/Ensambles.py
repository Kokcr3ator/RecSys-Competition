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



    def get_models_list(self):
        return self.recommenders_list
    

    
    def set_models_list(self, models_list):
        self.recommenders_list = models_list

    def set_weights_list(self, weights_list):
        self.weights_list = weights_list
        



    def fit(self, merge_topPop= False, topPop_factor= 1e-6):
        '''
            Fit each of the Recommender objects in the ensamble by calling fit() method for each of them.
        '''
        self.merge_topPop = merge_topPop

        # These parameters allow to utilize TopPopRecommender for filling in zero ratings, when you don't have enough
        # recommendations
        if self.merge_topPop:
            self.topPop_factor = topPop_factor


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

        if self.merge_topPop:
            n_items = self.URM_train.shape[1]

            # Compute TopPop
            item_popularity = np.ediff1d(self.URM_train.tocsc().indptr)
            popular_items = np.argsort(item_popularity)
            popular_items = np.flip(popular_items, axis = 0)
                
            # positions array is a vector containing the positions (from 1 to n_items)
            positions = np.arange(n_items)
            positions +=1

            # Create mapping to associate the position to the item_id
            map_index_position = {popular_items[i]:positions[i] for i in range(len(positions))}
            
            # Apply the column-wise operation : score = score + topPop_factor*(n_items - position)/ n_items
            def popularity_add(column, index):
                return column + self.topPop_factor*((n_items - map_index_position[index] )/(n_items)) 
                
            scores_batch = np.array([popularity_add(scores_batch[:, i], i) for i in range(n_items)]).T

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
        


    def set_URM_train(self, URM_train):
        self.URM_train = URM_train
        for recommender in self.recommenders_list:
            recommender.set_URM_train(URM_train)


class PipelineStep(BaseRecommender):
    """Recommender as step of a Recommenders' pipeline"""

    RECOMMENDER_NAME = "Pipeline_Step_Ensamble_Recommender_Class"

    def __init__(self, URM_input, recommender_object, hyperparameters_dict, n_relevant_per_user= 200, verbose=True):
        super(PipelineStep, self).__init__(URM_input, verbose=verbose)
        self.URM_train = None
        self.URM_input = URM_input

        self.recommender_object = recommender_object
        self.hyperparameters_dict = hyperparameters_dict

        self.n_relevant_per_user= n_relevant_per_user
        self.relevant_items_per_user = None
        self.relevant_items = None

        self.merge_topPop = False
        self.topPop_factor = 0.0


    
    def fit(self, merge_topPop= False, topPop_factor= 1e-6):
        # These parameters allow to utilize TopPopRecommender for filling in zero ratings, when you don't have enough
        # recommendations
        self.merge_topPop = merge_topPop
        if self.merge_topPop:
            self.topPop_factor = topPop_factor

        self.recommender_object.fit(**self.hyperparameters_dict)



    def recommend(self, cutoff = None, remove_zero_scores= True, return_scores = True):
        '''Custom recommend() method. No option for removing seen items (no point in deleting the ones in the URM_output).'''
        if cutoff is None:
            cutoff = self.URM_input.shape[1] - 1
        cutoff = min(cutoff, self.URM_input.shape[1] - 1)
        
        scores_batch = self.recommender_object._compute_item_score(np.range(self.n_users))

        if self.merge_topPop:
            n_items = self.URM_train.shape[1]

            # Compute TopPop
            item_popularity = np.ediff1d(self.URM_train.tocsc().indptr)
            popular_items = np.argsort(item_popularity)
            popular_items = np.flip(popular_items, axis = 0)
                
            # positions array is a vector containing the positions (from 1 to n_items)
            positions = np.arange(n_items)
            positions +=1

            # Create mapping to associate the position to the item_id
            map_index_position = {popular_items[i]:positions[i] for i in range(len(positions))}
            
            # Apply the column-wise operation : score = score + topPop_factor*(n_items - position)/ n_items
            def popularity_add(column, index):
                return column + self.topPop_factor*((n_items - map_index_position[index] )/(n_items)) 
                
            scores_batch = np.array([popularity_add(scores_batch[:, i], i) for i in range(n_items)]).T

        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index
        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = np.argpartition(-scores_batch, cutoff-1, axis=1)[:,0:cutoff] # get the partition for most relevant items

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        # Extract most relevant scores and indices
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition] 
        # Sort most relevant
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1) 
        # ranking -> each row contains the indices of the top cutoff items for the corresponding row in the original scores_batch
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(self.n_users):
            # user_recommendation_list -> indices of recommended items
            user_recommendation_list = ranking[user_index] 
            # user_item_scores -> scores corresponding to the recommended items
            user_item_scores = scores_batch[user_index, user_recommendation_list] 

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            if remove_zero_scores:
                zero_scores_mask = [user_item_scores <= 0.0] 
                not_inf_nor_zero_scores_mask = np.logical_or(not_inf_scores_mask, np.logical_not(zero_scores_mask))
                user_recommendation_list = user_recommendation_list[not_inf_nor_zero_scores_mask] 

            else: user_recommendation_list = user_recommendation_list[not_inf_scores_mask]

            ranking_list[user_index] = user_recommendation_list.tolist()


        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list
        

        
    def compute_relevant_items(self, at= 200):

        '''
        Computes the union of relevant items for all the users.
        Returns a mask n_items long with 1 if the item is relevant for at least 1 user, 0 otherwise

        '''
        

        relevant_items_all_users = self.recommend(self, cutoff = at, remove_zero_scores= True, return_scores = False)

        def relevant_item_mask_single_user(relevant_items_single_user):

            # Defining a function that for an array of relevant items of a user creates an array with length n_items 
            # with 1 if the item is relevant, 0 otherwise 

            mask = np.zeros(self.URM_train.shape[1])
            mask[relevant_items_single_user] = 1
        
            return mask
        
        # Matrix mask for each user relevant_items_mask(i,j) = 1 if item j is relevant for user i, 0 otherwise

        relevant_items_mask = np.array([relevant_item_mask_single_user(relevant_items_single_user) for relevant_items_single_user in relevant_items_all_users])

        # Compute the logical or between all the rows

        relevant_items = np.logical_or.reduce(relevant_items_mask)

        return relevant_items

    
    def compute_output_URM(self, remove_non_relevant_items= False, n_relevant_items_per_user= 200, remove_non_relevant_users= False):
        '''Produces a new URM by removing the non-relevant items or users for the model'''

        if remove_non_relevant_items:

            self.n_relevant_per_user = n_relevant_items_per_user

            # remove non relevant items

            items_to_keep = np.where(self.compute_relevant_items(at = self.n_relevant_per_user))[0]

            URM_inputcsc = self.URM_input.tocsc()

            self.URM_output = URM_inputcsc[:, items_to_keep.nonzero()[0]].tocsr()

            print("Successfully removed items non-relevant to the model.")

            return self.URM_output

        if remove_non_relevant_users:
            # remove non relevant users
            pass
            # TODO: self.URM_output = 
            #print("Successfully removed items non-relevant to the model.")



    def get_output_URM(self):
        if self.URM_output == None:
            print("Output URM has not been computed yet.\n Calling compute_output_URM().")
            self.compute_output_URM()
        else: return self.URM_output

    def get_relevant_items_per_user(self):
        if self.relevant_items_per_user == None:
            print("Relevant items to each user have not been computed yet.\n Calling compute_relevant_items().")
            self.compute_relevant_items()
        return self.relevant_items_per_user
    
    def get_relevant_items(self):
        if self.relevant_items == None:
            print("Relevant items have not been computed yet.\n Calling compute_relevant_items().")
            self.compute_relevant_items()
        return self.relevant_items



