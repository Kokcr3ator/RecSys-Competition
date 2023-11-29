#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.DataIO import DataIO
from Recommenders.NonPersonalizedRecommender import TopPop

import numpy as np



class BaseSimilarityMatrixRecommender(BaseRecommender):
    """
    This class refers to a BaseRecommender KNN which uses a similarity matrix, it provides two function to compute item's score
    bot for user-based and Item-based models as well as a function to save the W_matrix
    """

    def __init__(self, URM_train, verbose=True, merge_topPop= False, topPop_factor= 1e-6):
        super(BaseSimilarityMatrixRecommender, self).__init__(URM_train, verbose = verbose)

        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False

        # These parameters allow to utilize TopPopRecommender for filling in zero ratings, when you don't have enough
        # recommendations
        self.topPop_factor = 0.0
        if merge_topPop:
            self.topPop_factor = topPop_factor
        
        item_popularity = np.ediff1d(self.URM_train.tocsc().indptr)
        popular_items = np.argsort(item_popularity)
        popular_items = np.flip(popular_items, axis = 0)
        self.popular_items = popular_items



    def _check_format(self):

        if not self._URM_train_format_checked:

            if self.URM_train.getformat() != "csr":
                self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("URM_train", "csr"))

            self._URM_train_format_checked = True

        if not self._W_sparse_format_checked:

            if self.W_sparse.getformat() != "csr":
                self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("W_sparse", "csr"))

            self._W_sparse_format_checked = True




    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"W_sparse": self.W_sparse}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")



    #########################################################################################################
    ##########                                                                                     ##########
    ##########                               COMPUTE ITEM SCORES                                   ##########
    ##########                                                                                     ##########
    #########################################################################################################


class BaseItemSimilarityMatrixRecommender(BaseSimilarityMatrixRecommender):

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()
    
        if self.merge_topPop:
            n_items = self.URM_train.shape[1]
            
            # positions array is a vector containing the positions (from 1 to n_items)
            positions = np.arange(n_items)
            positions +=1

            # Create mapping to associate the position to the item_id
            map_index_position = {self.popular_items[i]:positions[i] for i in range(len(positions))}
        
            # Apply the column-wise operation : score = score + topPop_factor*(n_items - position)/ n_items
            def popularity_add(column, index):
                return column + self.topPop_factor*((n_items - map_index_position[index] )/(n_items)) 
            
            item_scores = np.array([popularity_add(item_scores[:, i], i) for i in range(n_items)]).T

        return item_scores


class BaseUserSimilarityMatrixRecommender(BaseSimilarityMatrixRecommender):

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_weights_array = self.W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            item_scores_all = user_weights_array.dot(self.URM_train).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_train).toarray()
    
            if self.merge_topPop:
                n_items = self.URM_train.shape[1]
                
                # positions array is a vector containing the positions (from 1 to n_items)
                positions = np.arange(n_items)
                positions +=1

                # Create mapping to associate the position to the item_id
                map_index_position = {self.popular_items[i]:positions[i] for i in range(len(positions))}
            
                # Apply the column-wise operation : score = score + topPop_factor*(n_items - position)/ n_items
                def popularity_add(column, index):
                    return column + self.topPop_factor*((n_items - map_index_position[index] )/(n_items)) 
                
                item_scores = np.array([popularity_add(item_scores[:, i], i) for i in range(n_items)]).T

        return item_scores
