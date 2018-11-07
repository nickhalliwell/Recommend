#!/usr/bin/env python3

import numpy as np
import pandas as pd
from collections import defaultdict
import netflixLoader as Loader
import json

def weightedAvgRatings(user_id, item_id,gamma, weightedAvgDict):
    '''
    Computes weighted average ratings for a user <user_id> and an item <item_id>

    Args:
    --------------------------------------------

    user_id: str, id of current user

    item_id: str, id of current item

    gamma: dict, 20 most similar users to current user, who have also rated current item
                 {user_id : {item_id_1:{sim_score_user_j, rating_item_i_user_j}, item_id_2 : {...}, ...}, user_id_2 : {...}, ...}

    weightedAvgDict: dict, weighted average ratings for current user and current item
                    {user_id : {item_id : weighted_avg_rating, ...}, ...}

    Output:
    --------------------------------------------

    weightedAvgDict: dict (see above)
    '''
    gammaUsers = gamma[user_id][item_id]

    sims = list(gammaUsers.keys())

    other_ratings = list(gammaUsers.values())

    if gammaUsers and sum(sims) > 0:

        weightedAvgRat = np.dot(sims, other_ratings) / len(gammaUsers) 

    else:

        weightedAvgRat = 0.0

    if user_id in weightedAvgDict:

        weightedAvgDict[user_id] = dict({item_id : weightedAvgRat},**weightedAvgDict[user_id])

    else:

        weightedAvgDict[user_id] = {item_id : weightedAvgRat}

    return weightedAvgDict


def updateDict(user_id, item_id, aspect, aspect_type,predAspDict, expression):
    '''
    Updates <predAspDict> with <expression>

    Args:
    --------------------------------------------

    user_id: str, id of current user

    item_id: str, id of current item

    aspect: str, current aspect to update

    aspect_type: str, current aspect_type to update

    predAspDict: dict, predicted aspect ratings for each user
                        {user_id : {item_id : {aspect_type : {aspect : expression}, ...}, item_id_2 : ...}, user_id_2 : ...}

    Output:
    --------------------------------------------

    predAspDict: dict
    '''
    if user_id in predAspDict:

        if item_id in predAspDict[user_id]:

            if aspect_type in predAspDict[user_id][item_id]:

                if aspect in predAspDict[user_id][item_id][aspect_type]:

                    predAspDict[user_id][item_id][aspect_type][aspect] = expression

                else:

                    predAspDict[user_id][item_id][aspect_type] = dict(predAspDict[user_id][item_id][aspect_type], **{aspect:expression})
            else:

                predAspDict[user_id][item_id] = dict(predAspDict[user_id][item_id], **{aspect_type :{aspect : expression}})

        else:

            predAspDict[user_id] = dict(predAspDict[user_id],**{item_id:{aspect_type :{aspect : expression}}})

    else:

        predAspDict[user_id] = {item_id:{aspect_type :{aspect : expression}}}
        
    return predAspDict
    

def avgRatingHelper(user_id,item_id, aspect,aspect_type, lmbda, weightedAvgDict, predAspDict):
    '''
    Updates <predAspDict> based on users in lmbda, lmbdaInv set

    Args:
    --------------------------------------------

    user_id: str, id of current user

    item_id: str, it of current item

    aspect: str, current aspect to update

    aspect_type: str, current aspect type to update

    lmbda: dict, items that have current aspect and item is rated by <user_id>
                {user_id : {aspect : {item_id : rating}, ...}, ...}

    weightedAvgDict: dict, weighted average ratings for current user and current item

    predAspDict: dict, predicted aspect ratings for each user

    Output:
    --------------------------------------------

    predAspDict: dict
    '''
    lmbdaInv = []

    for some_item, item_rating in weightedAvgDict[user_id].items():

        if item_rating and some_item not in lmbda[user_id][aspect]:

            lmbdaInv.append(item_rating)


    sum_lmbdaInv = (0.3 * sum(lmbdaInv))
    
    sum_lmbda = sum(lmbda[user_id][aspect].values())

    len_lmbda = len(lmbda[user_id][aspect])

    len_lmbdInv = len(lmbdaInv)

    denom = (1 + 0.3)

    if aspect in fullAspects[user_id][aspect_type]:

        predAspDict =  updateDict(user_id, item_id, aspect, aspect_type,predAspDict, fullAspects[user_id][aspect_type][aspect])

    else:

        if not lmbda[user_id][aspect] and not lmbdaInv:

            predAspDict =  updateDict(user_id, item_id, aspect, aspect_type,predAspDict, 0.0)

        else:

            if not lmbda[user_id][aspect]:

                expression = (sum_lmbdaInv / len_lmbdInv) / denom

                predAspDict =  updateDict(user_id, item_id, aspect, aspect_type,predAspDict, expression)

            else:

                if not lmbdaInv:

                    expression = sum_lmbda / len_lmbda

                    predAspDict = updateDict(user_id,item_id, aspect, aspect_type,predAspDict, expression)

                else:

                    expression = (((sum_lmbda) / len_lmbda) + (sum_lmbdaInv / len_lmbdInv)) / denom

                    predAspDict =  updateDict(user_id, item_id, aspect, aspect_type,predAspDict, expression)
    
    return predAspDict


def predAspectRating(user_id, item_id, weightedAvgDict, lmbda, fullAspects, predAspDict):
    '''
    Populates <predAspDict>

    Args:
    --------------------------------------------

    user_id: str, id of current user

    item_id: str, it of current item

    weightedAvgDict: dict, weighted average ratings for current user and current item

    lmbda: dict, items that have current aspect and item is rated by <user_id>

    fullAspects: dict, known aspect ratings for <user_id>

    predAspDict: dict, predicted aspect ratings for each user


    Output:
    --------------------------------------------

    predAspDict: dict

    '''
    item_aspects = itemAspectDict[item_id]

    for aspect_type, aspects in item_aspects.items():
        
        if aspect_type == 'title':
            
            continue

        elif isinstance(aspects, list):

            for aspect in aspects:

                predAspDict = avgRatingHelper(user_id,item_id, aspect,aspect_type, lmbda, weightedAvgDict,  predAspDict)
        else:
            predAspDict = avgRatingHelper(user_id,item_id, aspects,aspect_type, lmbda, weightedAvgDict, predAspDict)

    return predAspDict


def predItemRatings(user_id, item_id, row, weightedAvgDict,predAspDict,type_dict, predItemsDict):
    '''
    Computes predicted item rating and populates <predItemsDict>

    Args:
    --------------------------------------------

    user_id: str, id of current user

    item_id: str, it of current item

    row: pandas series, row of data

    weightedAvgDict: dict, weighted average ratings for current user and current item

    predAspDict: dict, predicted aspect ratings for each user

    type_dict: dict, type importance constant for each type
                    {aspect_type : type_constant, ...}

    predItemsDict: dict, predicted item rating for each user
                        {user_id : {item_id : pred_rating}, ...}


    Output:
    --------------------------------------------

    predItemsDict: dict
    '''
    subset = train_ratings[(train_ratings['userID'] == user_id) & (train_ratings['itemID'] == item_id)]
    
    sum_types = sum(type_dict.values())

    denom = (0.3 + sum_types)

    fraction = []

    for type_name, dictionary in predAspDict[user_id][item_id].items():

        scores = type_dict[type_name] * (sum(dictionary.values()) / len(dictionary.values()))

        fraction.append(scores)


    if not subset.empty and row.Index != subset.index:

        predItemsDict[user_id][item_id] = subset['rating'].values[0]

    else:

        if item_id in weightedAvgDict[user_id] and not sum_types:

            predItemsDict[user_id][item_id] =  weightedAvgDict[user_id][item_id]

        else:

            if item_id not in weightedAvgDict[user_id] and sum_types > 0:

                predItemsDict[user_id][item_id] = sum(fraction) / sum_types

            else:

                if item_id in weightedAvgDict[user_id] and (denom > 0):

                    predItemsDict[user_id][item_id] = ((0.3 * weightedAvgDict[user_id][item_id]) + sum(fraction)) / (denom)

                else:

                    predItemsDict[user_id][item_id] = 0.0
                        
    return predItemsDict

def RunModel(data, weightedAvgDict = None,predAspDict = None, Train=True):
    '''
    Iterates through <data>, computing predictions and computing accuracy

    Args:
    --------------------------------------------

    data: pandas dataframe, contains training/test data with columns [userID, itemID, rating]

    weightedAvgDict: dict, weighted average ratings for current user and current item

    predAspDict: dict, predicted aspect ratings for each user

    Train: bool, if model is running on training data (default True)


    Output:
    --------------------------------------------

    accuracy: float, accuracy of model on <data>

    weightedAvgDict: dict

    predAspDict: dict

    (if Train is False only accuracy is returned)
    '''

    #TYPE_DICT = {'genre' : 0.3,'actors' : 0.5, 'director' : 0.2}
    type_dict = {'Genre': 0.3, 'Author' : 0.5, 'Publisher': 0.2}

    if weightedAvgDict is None:

        weightedAvgDict = {}

    if predAspDict is None:

        predAspDict = {}

    predItemsDict = defaultdict(dict)

    output = 0

    for row in data.itertuples():

        if not row.Index % 100000 and row.Index:

            print('Current iteration',row.Index, 'Num correct', output)

        user_id = getattr(row, 'userID')
        item_id = getattr(row, 'itemID')
        true_rating = getattr(row, 'rating') 

        weightedAvgDict = weightedAvgRatings(user_id, item_id,gamma, weightedAvgDict)

        predAspDict = predAspectRating(user_id, item_id, weightedAvgDict,lmbda,fullAspects, predAspDict)

        predItemsDict = predItemRatings(user_id, item_id, row, weightedAvgDict,predAspDict,type_dict, predItemsDict)

        pred_rating = round(predItemsDict[user_id][item_id])

        #if (1 + pred_rating == true_rating) or (1 - pred_rating == true_rating) or (pred_rating == true_rating):
        if pred_rating == true_rating:

            output += 1
    
    accuracy = output / data.shape[0]

    if Train:

        return accuracy, weightedAvgDict, predAspDict


    return accuracy, predAspDict


#train_ratings, compressedTest, itemAspectDict,lmbda, gamma,fullAspects = Loader.Netflix(20,5)

################LOAD DATA HERE#################


###############################################


print("Data Loaded!")

train_accuracy, weightedAvgDict, predAspDict = RunModel(train_ratings)

print(train_accuracy)

test_accuracy, predAspDict = RunModel(compressedTest, 
                        weightedAvgDict = weightedAvgDict,
                        predAspDict = predAspDict,
                        Train=False)

print(test_accuracy)

