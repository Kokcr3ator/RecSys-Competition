from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.DataIO import DataIO
from Utils.write_ndarray_with_mask import write_ndarray_with_mask
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sps


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

            
    
    def _compute_item_score(self, user_id_array, items_to_compute = None):
        '''
        Compute the scores for the items provided by weighted-averaging over the scores computed by each of the 
        Recommenders in the Ensamble.

        '''
        # For each Recommender object we compute the scores and store them in scores_batch_array
        scores_batch_array = np.array([recommender_object._compute_item_score(user_id_array, items_to_compute=items_to_compute) 
                                       for recommender_object in self.recommenders_list])
        
        # Normalize each row (the scores attribted by each Recommender in the Ensamble)
        scaler = MinMaxScaler(feature_range=(-1, 1))

        for i in range(scores_batch_array.shape[0]):

            normalized_scores_rec_i = np.array([scaler.fit_transform(scores_batch_array[i][j,:].reshape(-1,1)).flatten() for j in range(scores_batch_array[i].shape[0])])
            scores_batch_array[i] = normalized_scores_rec_i
        
        # Now compute the ensamble scores by calculating a weighted average over the scores of each Recommender
        scores_batch = np.average(scores_batch_array, axis= 0, weights= self.weights_list)

        return scores_batch

    
 

            
    
    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):
        '''
        Compute the recommendations of the Ensamble

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

        # Map user_id_array and items_to_compute from original to preprocessed
        if self.manage_cold_items:
            OtP_item_mapping = pd.Series(self.PtO_item_mapping.index, index=self.PtO_item_mapping.values) # from orginal to preprocessed ID
            if items_to_compute is not None:
                items_to_compute = np.array(OtP_item_mapping.loc[items_to_compute].values) # preprocessed items id
        
        if self.manage_cold_users:
            # Distinguish between cold and non-cold (here referenced with 'hot') users
            hot_mask = np.array([np.isin(user_id, self.PtO_user_mapping.values) for user_id in user_id_array]) # check if user_id is in original ids
            cold_mask = np.logical_not(hot_mask)

            hot_users_id_array = user_id_array[hot_mask]

            # Compute cold users score using TopPop
            item_popularity = np.ediff1d(self.URM_train.tocsc().indptr)
            popular_items = np.argsort(item_popularity)
            popular_items = np.flip(popular_items, axis = 0)
            
            if self.manage_cold_items:
                popular_items = np.array(self.PtO_item_mapping.loc[popular_items].values) # map popular items to original IDs
                
            # positions array is a vector containing the positions (from 1 to n_items)
            positions = np.arange(len(popular_items))
            positions +=1

            # Create mapping to associate the position to the item_id
            map_index_position = {popular_items[i]:positions[i] for i in range(len(popular_items))}

            # Compute TopPop scores
            scores_topPop = - np.ones(self.n_items , dtype=np.float32) * np.inf
            for item_id in popular_items:
                scores_topPop[item_id] = (self.n_items - map_index_position[item_id])/(self.n_items)


            # Map hot users ids from original to preprocessed representation, in order to feed _compute_item_score()
            OtP_user_mapping = pd.Series(self.user_mapping.index, index = self.user_mapping.values)
            hot_users_id_array_preprocessed = np.array(OtP_user_mapping.loc[hot_users_id_array].values)

            # Compute scores for hot users using _compute_item_score() and fill the scores_batch
            scores_batch = - np.ones((len(user_id_array), self.n_items), dtype=np.float32) * np.inf
            if self.manage_cold_items:
                scores_batch = write_ndarray_with_mask(scores_batch, 
                                                       hot_mask, self.PtO_item_mapping.values, 
                                                       self._compute_item_score(hot_users_id_array_preprocessed, 
                                                                                items_to_compute=items_to_compute))
            else:
                scores_batch[hot_mask, :] = self._compute_item_score(hot_users_id_array_preprocessed, items_to_compute=items_to_compute)
            
            if np.all(hot_mask):
                pass
            else:
                scores_batch[cold_mask, :] = np.array([scores_topPop for i in range(np.sum(cold_mask))])

        else:
            # Compute the scores using the model-specific function
            # Vectorize over all users in user_id_array
            if self.manage_cold_items:
                scores_batch = - np.ones((len(user_id_array), self.n_items), dtype=np.float32) * np.inf
                scores_batch[:, self.PtO_item_mapping.values] = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)
            else:
                scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)

        if self.merge_topPop:
            if items_to_compute is not None:
                n_items = len(items_to_compute)
            else:
                n_items = self.n_items

            # Compute TopPop
            item_popularity = np.ediff1d(self.URM_train.tocsc().indptr)
            popular_items = np.argsort(item_popularity)
            popular_items = np.flip(popular_items, axis = 0)

            if self.manage_cold_items:
                popular_items = np.array(self.PtO_item_mapping.loc[popular_items].values) # map popular items to original IDs
                
            # positions array is a vector containing the positions (from 1 to n_items)
            positions = np.arange(len(popular_items))
            positions +=1

            # Create mapping to associate the position to the item_id
            map_index_position = {popular_items[i]:positions[i] for i in range(len(popular_items))}
            
            # Apply the column-wise operation : score = score + topPop_factor*(n_items - position)/ n_items
            def popularity_add(column, index):
                return column + self.topPop_factor*((n_items - map_index_position[index] )/(n_items)) 
            
            scores_batch_pop = np.array([popularity_add(scores_batch[:, item], item) for item in popular_items]).T
            scores_batch[:, popular_items] = scores_batch_pop 

        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if self.manage_cold_users and np.isin(user_id, hot_users_id_array):
                user_id = OtP_user_mapping.loc[user_id]

            if remove_seen_flag:
                scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

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
            
            
    
    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))
        
        for recommender_object in self.recommenders_list:
            recommender_object.save_model(folder_path = folder_path + "/" + self.RECOMMENDER_NAME)

        self._print("Saving complete")
        
    
    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        for recommender_object in self.recommenders_list:
            recommender_object.load_model(folder_path = folder_path)
        
        self._print("Loading complete")
            


    def set_URM_train(self, URM_train):
        self.URM_train = URM_train
        for recommender in self.recommenders_list:
            recommender.set_URM_train(URM_train)
        
        
      
    def get_models_list(self):
        return self.recommenders_list
   
    def set_models_list(self, models_list):
        self.recommenders_list = models_list
      
      

    def set_weights_list(self, weights_list):
        self.weights_list = weights_list
        
        
    
    def set_merge_topPop(self, merge_topPop):
        self.merge_topPop = merge_topPop
    
    def set_topPop_factor(self, topPop_factor):
        self.topPop_factor = topPop_factor


            

class PipelineStep(BaseRecommender):
    """Recommender as step of a Recommenders' pipeline"""

    RECOMMENDER_NAME = "Pipeline_Step_Ensamble_Recommender_Class"

    def __init__(self, URM_input, recommender_object, hyperparameters_dict= None, n_relevant_per_user= 200, verbose=True):
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
        if not self.hyperparameters_dict == None:
            self.recommender_object.fit(**self.hyperparameters_dict)
        else: print("No hyperparameters for fitting were provided. Check if the method is already fitted or provide an empty dict to use default hyperparameters.")


    def set_previous_pipelineStep_mapping(self, previous_pipelineStep_relevant_items):
        self.manage_previous_pipelineStep = True
        # from original to pipeline mapping
        self.pipelineStep_mapping = pd.Series(np.arange(len(previous_pipelineStep_relevant_items)), # pipeline
                                              index= previous_pipelineStep_relevant_items) # original
        self.n_items = len(self.previous_pipelineStep_relevant_items)
        


    def recommend(self, user_id_array= None, cutoff = None, remove_zero_scores= True, return_scores = True):
        '''Custom recommend() method. No option for removing seen items (no point in deleting the ones in the URM_output).'''
        if cutoff is None:
            cutoff = self.URM_input.shape[1] - 1
        cutoff = min(cutoff, self.URM_input.shape[1] - 1)     

        if user_id_array is None:
            user_id_array = np.arange(self.n_users)  
        
        if self.manage_previous_pipelineStep:
            if items_to_compute is not None:
                items_to_compute = np.array(self.pipelineStep_mapping.loc[items_to_compute].values) # pipeline items id

            scores_batch = - np.ones((len(user_id_array), self.n_items), dtype=np.float32) * np.inf
            scores_batch[:, items_to_compute] = self.recommender_object._compute_item_score(user_id_array, items_to_compute=items_to_compute)
        
        else: scores_batch = self.recommender_object._compute_item_score(user_id_array, items_to_compute=items_to_compute)

        if self.merge_topPop:
            n_items = self.URM_input.shape[1]

            # Compute TopPop
            item_popularity = np.ediff1d(self.URM_input.tocsc().indptr)
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
                non_zero_scores_mask = np.logical_not(user_item_scores <= 0.0)
                not_inf_nor_zero_scores_mask = np.logical_and(not_inf_scores_mask, non_zero_scores_mask)
                user_recommendation_list = user_recommendation_list[not_inf_nor_zero_scores_mask] 

            else: user_recommendation_list = user_recommendation_list[not_inf_scores_mask]

            ranking_list[user_index] = user_recommendation_list.tolist()


        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list

        
    def compute_relevant_items(self, at= 200, return_ids= False):

        '''
        Computes the union of relevant items for all the users.
        Returns a mask n_items long with 1 if the item is relevant for at least 1 user, 0 otherwise

        '''

        relevant_items_per_users = self.recommend(cutoff = at, remove_zero_scores= True, return_scores = False)
        self.relevant_items_per_user = relevant_items_per_users

        def relevant_item_mask_single_user(relevant_items_single_user):
            # Defining a function that for an array of relevant items of a user creates an array with length n_items 
            # with 1 if the item is relevant, 0 otherwise 

            mask = np.zeros(self.URM_input.shape[1])
            mask[relevant_items_single_user] = 1
        
            return mask
        
        # Matrix mask for each user relevant_items_mask(i,j) = 1 if item j is relevant for user i, 0 otherwise
        relevant_items_mask = np.array([relevant_item_mask_single_user(relevant_items_single_user) for relevant_items_single_user in relevant_items_all_users])

        # Compute the logical or between all the rows
        self.relevant_items_mask = np.logical_or.reduce(relevant_items_mask)

        if return_ids:
            relevant_sets = [set(row) for row in relevant_items_per_users]
            relevant_items_ids = set().union(*relevant_sets)
            relevant_items_ids = np.array(list(relevant_items_ids))
            return self.relevant_items_mask, relevant_items_ids
        else:
            return self.relevant_items_mask

    
    
    def compute_output_URM(self, remove_non_relevant_items= True, n_relevant_items_per_user= 200, remove_non_relevant_users= False):
        '''
        Produces a new URM by removing the non-relevant items or users for the model
        
        '''
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
        if self.relevant_items_per_user.any() == None:
            print("Relevant items to each user have not been computed yet.\n Calling compute_relevant_items().")
            self.compute_relevant_items()
        return self.relevant_items_per_user
    
    def get_relevant_items(self):
        if self.relevant_items.any() == None:
            print("Relevant items have not been computed yet.\n Calling compute_relevant_items().")
            self.compute_relevant_items()
        return self.relevant_items



class UserSpecific(LinearCombination):
    """User Specific Ensamble Recommender"""

    RECOMMENDER_NAME = "User_Specific_Ensamble_Recommender"

    def __init__(self, URM_train, recommenders_list, hyperparameters_dicts_list, user_groups, weights_list= None, original_URM_train = None, verbose = True):
        super(LinearCombination, self).__init__(URM_train, verbose = verbose)

        """ 
        PARAMETERS:

            MANDATORY:

            -URM_train: csr format User Rating Matrix 
            -recommenders_list: list of recommenders, one for each group
            -hyperparameters_dicts_list: list of dictionaries for the hyperparameters of the recommender in each group
            -user_groups: array of tuples [(a_0, a_1),(b_0, b_1),...] where each tuples consitutes a group, e.g:
                [(0,3),(3,6),(6,19)] -> 3 groups such that the first contains the first 20% of users in ascending order of number of interactions.
                The second group will be the users from 20 to 35-percentile  in ascending order of number of interactions.
                The last group contain the remaining users.
            
            OPTIONAL:

            -weights_list: list of lists of weights in case using LinearCombination ensambles for some groups
            -original_URM_train: csr format User Rating Matrix in case of using preprocessed URM
            -verbose: boolean for verbosity

        """
        
        assert len(recommenders_list) != len(hyperparameters_dicts_list) != len(user_groups),\
        "recommenders_list, hyperparameters_dicts_list, weights_list and user_groups should all have the same length. Lengths provided: {}, {}, {}.".format(
        len(recommenders_list), len(hyperparameters_dicts_list), len(user_groups) )

        self.recommenders_list = recommenders_list # list of initialized recommenders
        self.hyperparameters_dicts_list = hyperparameters_dicts_list
        self.weights_list = weights_list

        assert user_groups[-1][1] != 19,\
        "Last element of the last tuple in user_groups should be 19"

        self.user_groups = user_groups

        if original_URM_train is not None:
            self.original_URM_train = original_URM_train
        else:
            self.original_URM_train = self.URM_train

        
        profile_length = np.ediff1d(sps.csr_matrix(self.original_URM_train).indptr)
        block_size = int(len(profile_length)*0.05)
        sorted_users = np.argsort(profile_length)
        grouped_users = [sorted_users[group[0]*block_size : group[1]*block_size] for group in self.user_groups]
        # For the last group I actually need to put all the users until the last one
        grouped_users[-1] = sorted_users[self.user_groups[-1][0]*block_size : len(profile_length)]
        # Convert arrays to sets for faster membership checking
        self.users_sets = [set(users) for users in self.grouped_users]

          

    def fit(self, merge_topPop= False, topPop_factor= 1e-6):

        """
            Fit each of the Recommender provided by calling fit() method for each of them,
            also sets the weights list in case of an ensamble.

        """
        self.merge_topPop = merge_topPop

        # These parameters allow to utilize TopPopRecommender for filling in zero ratings, when you don't have enough
        # recommendations
        if self.merge_topPop:
            self.topPop_factor = topPop_factor


        for i in range(len(self.recommenders_list)):
            hyperparams = self.hyperparameters_dicts_list[i]
            self.recommenders_list[i].fit(**hyperparams)
            print("Successfully fitted Recommender: ", self.recommenders_list[i].RECOMMENDER_NAME)
            if self.recommenders_list[i] == "Linear_Combination_Ensamble_Recommender_Class" :
                self.recommenders_list[i].set_weights_list(self.weights_list[i])
                print("Successfully set weights for LinearCombination Recommender")
    


    def find_group(self, user):

        """ 
        Given a users returns the group he belongs to, in case a user is not present 
        in the sets (cold_users) it will return group 0 (least interactions group)

        """
        for index, array_set in enumerate(self.users_sets):
            if user in array_set:
                return index
        
        return 0  
            

    def assign_group_to_user_id_array(self, user_id_array):

        # Use multiprocessing to parallelize computations

        with Pool() as pool:
            group_assignments = pool.map(self.find_group, user_id_array)

        return group_assignments

                
    def _compute_item_score(self, user_id_array, items_to_compute = None):

        raise NotImplementedError("compute_item_score has not been implemented.")

            
    
    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False):
        '''
        Compute the recommendations of the Ensamble

        '''

        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False
        
        # Assign the group to each user
        group_assignments = np.array(self.assign_group_to_user_id_array(user_id_array))

        if self.original_URM_train is not None:
            n_items = self.original_URM_train.shape[1]
        else:
            n_items = self.URM_train.shape[1]
        
        n_users = len(user_id_array)

        ranking_list_array = np.zeros((n_users,n_items))

        for i in range(len(self.recommenders_list)):
            recommender_mask = (np.where(group_assignments == i, 1,0)).astype(bool)
            recommendations_lists = self.recommenders_list[i].recommend(user_id_array[recommender_mask].tolist(),
                                                                        cutoff = cutoff,
                                                                        remove_seen_flag = remove_seen_flag,
                                                                        items_to_compute = items_to_compute,
                                                                        remove_top_pop_flag = remove_top_pop_flag,
                                                                        remove_custom_items_flag = remove_custom_items_flag,
                                                                        return_scores = False)
            recommendations_array = np.array([np.array(recommendations) for recommendations in recommendations_lists])
            ranking_list_array[recommender_mask] = recommendations_array

        # Transform back to list of lists
        ranking_list = ranking_list_array.tolist()

        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        return ranking_list
            
            
    
    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))
        
        for recommender_object in self.recommenders_list:
            recommender_object.save_model(folder_path = folder_path + "/" + self.RECOMMENDER_NAME)

        self._print("Saving complete")
        
    
    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        for recommender_object in self.recommenders_list:
            recommender_object.load_model(folder_path = folder_path)
        
        self._print("Loading complete")
            


    def set_URM_train(self, URM_train):
        self.URM_train = URM_train
        for recommender in self.recommenders_list:
            recommender.set_URM_train(URM_train)
        
