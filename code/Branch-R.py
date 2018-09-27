#!/usr/bin/env python3

import pickle
import os
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import netflixLoader as Loader

#os.chdir('..')

def weightedAvgRatings(data, user_id, item_id,sim_dict, weightedAvgDict):

    #gammaUsers = train_ratings[(train_ratings['itemID'] == item_id) & (train_ratings['userID'] != user_id)]
    gammaUsers = data[(data['itemID'] == item_id) & (data['userID'] != user_id)]

    numUsers = gammaUsers.shape[0]

    sims = []

    for gamma_row in gammaUsers.itertuples():

        other_user = getattr(gamma_row, 'userID')

        get_sims = sim_dict[user_id].get(other_user, sim_dict[other_user].get(user_id, 0.0))

        sims.append(get_sims)

    if not gammaUsers.empty and sum(sims) > 0:

        weightedAvgRat = np.dot(sims, gammaUsers['rating']) / numUsers

    #else:

        #weightedAvgRat = 0.0

        if user_id in weightedAvgDict:

            weightedAvgDict[user_id] = dict({item_id : weightedAvgRat},**weightedAvgDict[user_id])

        else:

            weightedAvgDict[user_id] = {item_id : weightedAvgRat}

    return weightedAvgDict


def updateDict(user_id, aspect, aspect_type,predAvgDict, expression):
    
    if user_id in predAvgDict:
        
        if aspect_type in predAvgDict[user_id]:
            
            if aspect in predAvgDict[user_id][aspect_type]:
                
                predAvgDict[user_id][aspect_type][aspect] = expression
                
            else:
                
                predAvgDict[user_id][aspect_type] = dict(predAvgDict[user_id][aspect_type],**{aspect : expression})
        else:
            
            predAvgDict[user_id] = dict(predAvgDict[user_id],**{aspect_type :{aspect : expression}})
    else:
        
        predAvgDict[user_id] = {aspect_type :{aspect : expression}}
        
    return predAvgDict
    

def avgRatingHelper(user_id,aspect,aspect_type, lmbda,lmbdaInv, predAvgDict):
    
    if aspect in lmbda:
        
        predAvgDict =  updateDict(user_id, aspect, aspect_type,predAvgDict, lmbda[aspect])
            
    else:
        
        if not lmbda and not lmbdaInv:
            
            predAvgDict =  updateDict(user_id, aspect, aspect_type,predAvgDict, 0.0)

        else:

            sum_lmbdaInv = (0.3 * (sum(lmbdaInv) / len(lmbdaInv)))

            sum_lmbda = sum(lmbda.values()) / len(lmbda.values())           
            
            if not lmbda:
                
                predAvgDict =  updateDict(user_id, aspect, aspect_type,predAvgDict, (sum_lmbdaInv / (1 + 0.3)))

                        
            else:
                                
                if not lmbdaInv:
                    
                    predAvgDict = updateDict(user_id, aspect, aspect_type,predAvgDict, sum_lmbda)
                        
                else:
                    
                    predAvgDict =  updateDict(user_id, aspect, aspect_type,predAvgDict, ((sum_lmbda + sum_lmbdaInv) / (1 + 0.3)) )
    
    return predAvgDict

def predAspectRating(data, user_id, item_id, weightedAvgDict, fullAspects, predAvgDict):

    #current item aspects
    item_aspects = itemAspectDict[item_id]

    #dictionary of {user_i: {aspect:rating}} of item i that have been rated by user i
    lmbda = {}#defaultdict(dict)

    for aspect_type, aspect in item_aspects.items():

    	if isinstance(aspect, list):

    		for element in aspect:

    			lmbda[element] = fullAspects[user_id][aspect_type][element]

    	elif aspect_type == 'title':

    		continue

    	else:

             lmbda[aspect] = fullAspects[user_id][aspect_type][aspect]


    lmbdaInv = []

    for not_user_i,d in weightedAvgDict.items():

        if not_user_i != user_id:

            for other_item, avg_rating in d.items():

                if avg_rating and other_item not in lmbda:

                    lmbdaInv.append(avg_rating)


    for aspect_type, aspects in item_aspects.items():
        
        if aspect_type == 'title':
            
            continue

        elif isinstance(aspects, list):

            for aspect in aspects:

                predAvgDict = avgRatingHelper(user_id,aspect,aspect_type, lmbda,lmbdaInv, predAvgDict)
        else:
            predAvgDict = avgRatingHelper(user_id,aspects,aspect_type, lmbda,lmbdaInv, predAvgDict)

    return predAvgDict


def predItemRatings(data, user_id, item_id, row, weightedAvgDict,predAvgDict,typeDict, predItemsDict):

    #find if user has already rated item i in training set
    #subset = train_ratings[(train_ratings['userID'] == user_id) & (train_ratings['itemID'] == item_id)]
    subset = data[(data['userID'] == user_id) & (data['itemID'] == item_id)]

    if not subset.empty and row.Index != subset.index:

        predItemsDict[user_id][item_id] = subset['rating'].values[0]

    else:

        sum_types = sum(typeDict.values())

        if item_id in weightedAvgDict[user_id] and not sum_types:

            predItemsDict[user_id][item_id] =  weightedAvgDict[user_id][item_id]

        else:


            fraction = []

            for type_name, dictionary in predAvgDict[user_id].items():

                    scores = typeDict[type_name] * (sum(dictionary.values()) / len(dictionary.values()))

                    fraction.append(scores)


            if item_id not in weightedAvgDict[user_id] and sum_types > 0:

                predItemsDict[user_id][item_id] = sum(fraction) / sum_types

            else:

                denom = (0.3 + sum_types)

                if item_id in weightedAvgDict[user_id] and (denom > 0):

                    predItemsDict[user_id][item_id] = ((0.3 * weightedAvgDict[user_id][item_id]) + sum(fraction)) / (denom)

                else:

                    predItemsDict[user_id][item_id] = 0.0
                        
    return predItemsDict

def RunModel(data, sim_dict, weightedAvgDict = {},predAvgDict = {}, Train=True):

    predItemsDict = defaultdict(dict)

    typeDict = {'genre' : 0.3,'actors' : 0.5, 'director' : 0.2}

    output = []

    for row in data.itertuples():

        user_id = getattr(row, 'userID')
        item_id = getattr(row, 'itemID')
        true_rating = getattr(row, 'rating') 

        weightedAvgDict = weightedAvgRatings(data, user_id, item_id, sim_dict, weightedAvgDict)

        predAvgDict = predAspectRating(data, user_id, item_id, weightedAvgDict, fullAspects, predAvgDict)

        predItemsDict = predItemRatings(data, user_id, item_id, row, weightedAvgDict,predAvgDict,typeDict, predItemsDict)

        pred_rating = round(predItemsDict[user_id][item_id])

        if (1 + pred_rating == true_rating) or (1 - pred_rating == true_rating) or (pred_rating == true_rating):

            output.append(1)
            
        else:
            
            output.append(0)
    
    accuracy = sum(output) / len(output)

    if Train:

        return accuracy, weightedAvgDict, predAvgDict

    return accuracy


train_ratings, itemAspectDict, userAspectDict, trainSimDict = Loader.NetflixTrain(20, 5)

compressedTest,testUserAspectDict,testSimsDict = Loader.NetflixTest(20, 5)

print("Data Loaded!")

fullSims = dict(trainSimDict, **testSimsDict)
fullAspects = dict(userAspectDict, **testUserAspectDict)

train_accuracy, weightedAvgDict, predAvgDict = RunModel(train_ratings, 
                                                        sim_dict = fullSims, 
                                                        weightedAvgDict = {},
                                                        predAvgDict = {},
                                                        Train = True)

print(train_accuracy)

test_accuracy = RunModel(compressedTest,
                        sim_dict = fullSims, 
                        weightedAvgDict = weightedAvgDict,
                        predAvgDict = predAvgDict,
                        Train=False)

print(test_accuracy)





