#!/usr/bin/env python3

import os
import pandas as pd
from collections import defaultdict
import pickle

os.chdir('..')

# def NetflixTrain(min_nr_movies_train, min_nr_movies_test, surprise=False):

# 	trainRatingsDict = pickle.load(open("imperial_ijcai/test_%s_%s/train_ratings_dict.pkl"  % (min_nr_movies_train, min_nr_movies_test), "rb"))

# 	train_ratings = pd.DataFrame.from_dict(trainRatingsDict)

# 	train_ratings['rating'] = train_ratings['rating'].astype(float)

# 	if surprise:

# 		return train_ratings

# 	itemAspectDict = pickle.load(open("imperial_ijcai/db/movie_metadata.pkl","rb"))

# 	all_actors = pickle.load(open("imperial_ijcai/arg/preference_actors_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))

# 	all_directors = pickle.load(open("imperial_ijcai/arg/preference_director_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))

# 	all_genres = pickle.load(open("imperial_ijcai/arg/preference_genre_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))

# 	trainSimDict = pickle.load(open("imperial_ijcai/arg/similarity_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))


# 	itemAspectDictReversed = defaultdict(list)

# 	for movie_id, dictionary in itemAspectDict.items():

# 		for aspect_type, aspect in dictionary.items():

# 			if aspect_type == 'title':

# 				continue

# 			elif isinstance(aspect, list):

# 				for a in aspect:

# 					itemAspectDictReversed[a].append(movie_id)

# 			else:

# 				itemAspectDictReversed[aspect].append(movie_id)


# 	userAspectDict = defaultdict(dict)

# 	for uid, dictionary in all_genres.items():
	        
# 	    userAspectDict[uid]['genre'] = dictionary
	    
# 	    userAspectDict[uid]['actors'] = all_actors[uid]
	    
# 	    userAspectDict[uid]['director'] = all_directors[uid]


# 	return train_ratings.head(), itemAspectDict, itemAspectDictReversed, userAspectDict, trainSimDict

# def NetflixTest(min_nr_movies_train, min_nr_movies_test, surprise=False):

# 	compressed_test_ratings_dict = pd.read_pickle("imperial_ijcai/test_%s_%s/compressed_test_ratings_dict.pkl"  
# 	 																		% (min_nr_movies_train, min_nr_movies_test))

# 	testing_users_cold_start = pd.read_pickle("imperial_ijcai/test_%s_%s/testing_users_cold_start.pkl"  
# 																		% (min_nr_movies_train, min_nr_movies_test))
	
# 	userIDtest = []
# 	itemIDtest = []
# 	ratingTest = []

# 	for user_id, true_ratings in compressed_test_ratings_dict.items():

# 		if true_ratings:

# 			for film_id, str_rating in true_ratings:

# 				userIDtest.append(user_id)
# 				itemIDtest.append(film_id)
# 				ratingTest.append(float(str_rating))

# 	compressedTest = pd.DataFrame({'userID' : userIDtest, 'rating':ratingTest,'itemID' : itemIDtest})

# 	if surprise:

# 		return compressedTest
		
# 	testUserAspectDict = defaultdict(dict)
# 	#testUserAspectDict = {}

# 	testSimsDict = {}

# 	for user_id, d in testing_users_cold_start.items():

# 		testUserAspectDict[user_id]['director'] = d['directors'][user_id]
# 		testUserAspectDict[user_id]['actors'] = d['actors'][user_id]
# 		testUserAspectDict[user_id]['genre'] = d['genres'][user_id]

# 		testSimsDict[user_id] = dict(d['sims'])

	
# 	return compressedTest.head(),testUserAspectDict,testSimsDict

