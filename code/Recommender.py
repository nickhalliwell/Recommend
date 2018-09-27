#!/usr/bin/env python3

import pickle
import os
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import netflixLoader as Loader

#os.chdir('..')

def weightedAvgRatings(data,sim_dict):

    weightedAvgDict = {}

    for row in data.itertuples():

        user_id = getattr(row, 'userID')
        item_id = getattr(row, 'itemID')

        gammaUsers = train_ratings[(train_ratings['itemID'] == item_id) & (train_ratings['userID'] != user_id)]

        numUsers = gammaUsers.shape[0]

        sims = []

        for gamma_row in gammaUsers.itertuples():

            other_user = getattr(gamma_row, 'userID')

            get_sims = sim_dict[user_id].get(other_user, sim_dict[other_user].get(user_id, 0.0))

            sims.append(get_sims)

        if not gammaUsers.empty and sum(sims) > 0:

            weightedAvgRat = np.dot(sims, gammaUsers['rating']) / numUsers

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
            
            frac = (0.3 * sum(lmbdaInv) / len(lmbdaInv)) 
            
            if not lmbda:
                
                predAvgDict =  updateDict(user_id, aspect, aspect_type,predAvgDict, (frac / (1 + 0.3)))

                        
            else:
                
                sum_lmbdaInv = sum(lmbdaInv) / len(lmbdaInv)
                
                if not lmbdaInv:
                    
                    predAvgDict = updateDict(user_id, aspect, aspect_type,predAvgDict, sum_lmbdaInv)
                        
                else:
                    
                    predAvgDict =  updateDict(user_id, aspect, aspect_type,predAvgDict, ((sum_lmbdaInv + frac) / (1 + 0.3)))
    
    return predAvgDict

def predAspectRating(data, weightedAvgDict, userAspectDict):

	predAvgDict = {}

	for row in data.itertuples():

	    user_id = getattr(row, 'userID')
	    item_id = getattr(row, 'itemID')    

	    #current item aspects
	    item_aspects = itemAspectDict[item_id]

	    #dictionary of {user_i: {aspect:rating}} of item i that have been rated by user i
	    lmbda = defaultdict(dict)

	    lmbdaInv = []

	    for aspect_type, aspect in item_aspects.items():

	    	if isinstance(aspect, list):

	    		for element in aspect:

	    			lmbda[element] = userAspectDict[user_id][aspect_type][element]

	    	elif aspect_type == 'title':

	    		continue

	    	else:

                 lmbda[aspect] = userAspectDict[user_id][aspect_type][aspect]


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


def predItemRatings(data, weightedAvgDict,predAvgDict,typeDict):
    
    predItemsDict = defaultdict(dict)

    for row in data.itertuples():

            user_id = getattr(row, 'userID')
            item_id = getattr(row, 'itemID')

            current_item_aspects = itemAspectDict[item_id]

            #find if user has already rated item i in training set
            subset = train_ratings[(train_ratings['userID'] == user_id) & (train_ratings['itemID'] == item_id)]
            #subset = data[(data['userID'] == user_id) & (data['itemID'] == item_id)]

            if not subset.empty and row.Index != subset.index:

                predItemsDict[user_id][item_id] = subset['rating'].values[0]

            else:

                #sum_types = sum(typeDict[user_id].values())
                sum_types = sum(typeDict.values())

                if item_id in weightedAvgDict[user_id] and not sum_types:

                    predItemsDict[user_id][item_id] =  weightedAvgDict[user_id][item_id]

                else:


                    fraction = []

                    for type_name, dictionary in predAvgDict[user_id].items():

                            #scores = typeDict[user_id][type_name] * (sum(dictionary.values()) / len(dictionary.values()))
                            scores = typeDict[type_name] * (sum(dictionary.values()) / len(dictionary.values()))

                            fraction.append(scores)


                    if item_id not in weightedAvgDict[user_id] and sum_types > 0:

                        predItemsDict[user_id][item_id] = sum(fraction) / sum_types

                    else:

                        denom = (0.3 + sum_types)

                        if item_id in weightedAvgDict[user_id] and (denom > 0):

                            predItemsDict[user_id][item_id] = ((0.3 *weightedAvgDict[user_id][item_id]) + sum(fraction)) / (denom)

                        else:

                            predItemsDict[user_id][item_id] = 0.0
                            
    return predItemsDict

def getAccuracy(data, predItemsDict):
    
    output = []

    #predictions = defaultdict(dict)
    
    for row in data.itertuples():
        
        user_id = getattr(row, 'userID')
        item_id = getattr(row, 'itemID')   
        true_rating = getattr(row, 'rating') 

        pred_rating = round(predItemsDict[user_id][item_id])
        
        #predictions[user_id][item_id] = pred_rating
        
        if (1 + pred_rating == true_rating) or (1 - pred_rating == true_rating) or (pred_rating == true_rating):

            output.append(1)
            
        else:
            
            output.append(0)
    
    accuracy = sum(output) / len(output)
    
    return accuracy#, predictions




train_ratings, itemAspectDict, userAspectDict, trainSimDict = Loader.NetflixTrain(10, 5)

compressedTest,testUserAspectDict,testSimsDict = Loader.NetflixTest(10, 5)

print("Data Loaded!")

fullSims = dict(trainSimDict, **testSimsDict)
fullAspects = dict(userAspectDict, **testUserAspectDict)
typeDict = {'genre' : 0.3,'actors' : 0.5, 'director' : 0.2}

weightedAvgDictTrain = weightedAvgRatings(train_ratings,fullSims)
print('weightedAvgDictTrain Done!')
#predAvgDict = predAspectRating(train_ratings,weightedAvgDictTrain,userAspectDict)
predAvgDict = predAspectRating(train_ratings,weightedAvgDictTrain,fullAspects)

print('predAvgDictTrain Done!')

predItemsDictTrain = predItemRatings(train_ratings, weightedAvgDictTrain,predAvgDict)
print('predItemsDictTrain Done!')
#train_acc, trainPreds = getAccuracy(train_ratings, predItemsDictTrain)
train_acc = getAccuracy(train_ratings, predItemsDictTrain)

print(train_acc)




weightedAvgDictTest = weightedAvgRatings(compressedTest,fullSims)
weightedAvgDictTest = dict(weightedAvgDictTrain, **weightedAvgDictTest)
print('weightedAvgDictTest Done!')
#predAvgDictTest = predAspectRating(compressedTest,weightedAvgDictTest,testUserAspectDict)
predAvgDictTest = predAspectRating(compressedTest,weightedAvgDictTest,fullAspects)
predAvgDictTest = dict(predAvgDict, **predAvgDictTest)
print('predAvgDictTest Done!')

predItemsDictTest = predItemRatings(compressedTest, weightedAvgDictTest,predAvgDictTest,typeDict)
print('predItemsDictTest Done!')

#test_acc, testPreds = getAccuracy(compressedTest, predItemsDictTest)
test_acc = getAccuracy(compressedTest, predItemsDictTest)

print(test_acc)

