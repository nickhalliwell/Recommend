#!/usr/bin/env python3

import os
import pandas as pd
from collections import defaultdict
import pickle
import json

os.chdir('..')

def Netflix(min_nr_movies_train, min_nr_movies_test, surprise=False):

	trainRatingsDict = pickle.load(open("imperial_ijcai/test_%s_%s/train_ratings_dict.pkl"  % (min_nr_movies_train, min_nr_movies_test), "rb"))

	train_ratings = pd.DataFrame.from_dict(trainRatingsDict)

	train_ratings['rating'] = train_ratings['rating'].astype(float)

	compressed_test_ratings_dict = pd.read_pickle("imperial_ijcai/test_%s_%s/compressed_test_ratings_dict.pkl"
																	% (min_nr_movies_train, min_nr_movies_test))
	userIDtest = []
	itemIDtest = []
	ratingTest = []

	for user_id, true_ratings in compressed_test_ratings_dict.items():

		if true_ratings:

			for film_id, str_rating in true_ratings:

				userIDtest.append(user_id)
				itemIDtest.append(film_id)
				ratingTest.append(float(str_rating))

	compressedTest = pd.DataFrame({'userID' : userIDtest, 'rating':ratingTest,'itemID' : itemIDtest})


	if surprise:

		return train_ratings, compressedTest


	testing_users_cold_start = pd.read_pickle("imperial_ijcai/test_%s_%s/testing_users_cold_start.pkl"  
																		% (min_nr_movies_train, min_nr_movies_test))
	
	itemAspectDict = pickle.load(open("imperial_ijcai/db/movie_metadata.pkl","rb"))

	all_actors = pickle.load(open("imperial_ijcai/arg/preference_actors_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))

	all_directors = pickle.load(open("imperial_ijcai/arg/preference_director_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))

	all_genres = pickle.load(open("imperial_ijcai/arg/preference_genre_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))

	trainSimDict = pickle.load(open("imperial_ijcai/arg/similarity_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))


	itemAspectDictReversed = defaultdict(list)

	for movie_id, dictionary in itemAspectDict.items():

		for aspect_type, aspect in dictionary.items():

			if aspect_type == 'title':

				continue

			elif isinstance(aspect, list):

				for a in aspect:

					itemAspectDictReversed[a].append(movie_id)

			else:

				itemAspectDictReversed[aspect].append(movie_id)


	userAspectDict = defaultdict(dict)

	for uid, dictionary in all_genres.items():
	        
	    userAspectDict[uid]['genre'] = dictionary
	    
	    userAspectDict[uid]['actors'] = all_actors[uid]
	    
	    userAspectDict[uid]['director'] = all_directors[uid]


	testUserAspectDict = defaultdict(dict)

	testSimsDict = {}

	for user_id, d in testing_users_cold_start.items():

		testUserAspectDict[user_id]['director'] = d['directors'][user_id]
		testUserAspectDict[user_id]['actors'] = d['actors'][user_id]
		testUserAspectDict[user_id]['genre'] = d['genres'][user_id]

		testSimsDict[user_id] = dict(d['sims'])	 




	fullSims = dict(trainSimDict, **testSimsDict)

	fullAspects = dict(userAspectDict, **testUserAspectDict)



	fullData = pd.concat([train_ratings, compressedTest], axis=0)



	lmbda = defaultdict(dict)

	gamma = defaultdict(dict)

	for row in fullData.itertuples():

		user_id = getattr(row, 'userID')
		item_id = getattr(row, 'itemID')

		item_aspects = itemAspectDict[item_id]

		item_list = []

		for aspect_type, aspect in item_aspects.items():

			if isinstance(aspect, list):

				for element in aspect:

					linked_items = itemAspectDictReversed[element]

					lmbda_user_i = train_ratings[(train_ratings['itemID'].isin(linked_items)) & (train_ratings['userID'] == user_id)]

					lmbda[user_id][element] = dict(zip(lmbda_user_i['itemID'].values, lmbda_user_i['rating'].values))

			elif aspect_type == 'title':

				continue

			else:

				linked_items = itemAspectDictReversed[aspect]

				lmbda_user_i = train_ratings[(train_ratings['itemID'].isin(linked_items)) & (train_ratings['userID'] == user_id)]

				lmbda[user_id][aspect] = dict(zip(lmbda_user_i['itemID'].values, lmbda_user_i['rating'].values))



		usersRatedItemI = train_ratings[(train_ratings['itemID'] == item_id) & (train_ratings['userID'] != user_id)]

		out = []
		for gammaRow in usersRatedItemI.itertuples():

			gamma_user = getattr(gammaRow, 'userID')
			gamma_rating = getattr(gammaRow, 'rating')

			sim_val = fullSims[user_id].get(gamma_user, fullSims[gamma_user].get(user_id, 0.0))

			out.append((sim_val, gamma_rating))

		gammaUsers = sorted(out, reverse=True)[0:20]

		gamma[user_id][item_id] = dict(gammaUsers)

	return train_ratings, compressedTest, itemAspectDict,lmbda, gamma,fullAspects



'''itemAspectDict: {item_id_1: {'director': 'Peter Segal','genre': ['Comedy'],
							'actors': ['Jack Nicholson','Adam Sandler','Marisa Tomei','Woody Harrelson','John Turturro'],
							'title': 'Anger Management'}, item_id_2 : ...}'''







