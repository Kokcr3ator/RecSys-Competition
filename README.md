# RecSys-Competition
## How to submit:
In the TopPopRecommender.ipynb file, at the end of the script there is a function called write_predictions() which takes as inputs:  
 - a recommender object (which has to be fitted using .fit() before calling the function)
 - URM
 - user_original_ID_to_index which is a pandas series used to map the original user ID to the indexes from 0 to the length of the users
 - index_to_item_original, analogously to user_original_ID_to_index is a pandas series but does the inverse mapping (from the indexes from 0 to the length of the items to the original items id)

When called, write_predicitions() will output submission.csv ready to be submitted.
