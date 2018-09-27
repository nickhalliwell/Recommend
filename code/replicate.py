#!/usr/bin/env python3

import pickle
import os
import numpy as np
import pandas as pd
import netflixLoader as Loader
import os

#os.chdir('../Recommend')

#dictionary {movie id 1: [{'user_rating': '4', 'user_rating_date': '2005-07-05', 'user_id': '1380819'}, ...], movie id 2: ...}
#ratings = pickle.load(open("imperial_ijcai/db/movie_ratings_500_id.pkl","rb"))

#dictionary {movie id 1: {'director': 'Peter Segal', 'genre': ['Comedy'], 
						#'actors': ['Jack Nicholson', 'Adam Sandler', 'Marisa Tomei', 'Woody Harrelson', 'John Turturro'],
						# 'title': 'Anger Management'}, movie id 2: ...}
films = pickle.load(open("imperial_ijcai/db/movie_metadata.pkl","rb"))

min_nr_movies_train = 10
min_nr_movies_test = 5
#dictionary {user id: {'Sport': 3.5, 'War': 3.3333333333333335, 'History': 3.0, 'Action': 3.1578947368421053, ...}}
preferences = pickle.load(open("imperial_ijcai/arg/preference_genre_False_normalized_%s.pkl" % min_nr_movies_train,"rb"))

#dictionary {user id: {actor 1: 1.0, actor 2: 5.0, ...}}
all_actors = pickle.load(open("imperial_ijcai/arg/preference_actors_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))

#dictionary {user id: {director 1: 1.0, director 2: 5.0, ...}}
all_directors = pickle.load(open("imperial_ijcai/arg/preference_director_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))

#dictionary {user id: {'Sport': 3.5, 'War': 3.3333333333333335, 'History': 3.0, 'Action': 3.1578947368421053, ...}}
all_genres = pickle.load(open("imperial_ijcai/arg/preference_genre_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))

#dictionary {user id 1: {user id 2: 0.5}, user id 2: {user id 1, 0.5}}
all_similarities = pickle.load(open("imperial_ijcai/arg/similarity_False_normalized_%s.pkl" % min_nr_movies_train, "rb"))

######################################################

#dictionary {user id: [user 1, user 2], rating: [user 1 rating, user 2 rating], itemID : [movie 1 id, movie 2 id]}
#train_ratings_dict = pickle.load(open("imperial_ijcai/test_%s_%s/train_ratings_dict.pkl"  % (min_nr_movies_train, min_nr_movies_test), "rb"))

#dictionary {user id: [user 4, user 5], rating: [user 4 rating, user 5 rating], itemID : [movie 4 id, movie 5 id]}
#test_ratings_dict = pickle.load(open("imperial_ijcai/test_%s_%s/test_ratings_dict.pkl"  % (min_nr_movies_train, min_nr_movies_test), "rb"))

#dictionary {user id 6 : [(movie id, rating)], user id 7 : [(movie id, rating)], ...}
#compressed_test_ratings_dict = pickle.load(open("imperial_ijcai/test_%s_%s/compressed_test_ratings_dict.pkl"  % (min_nr_movies_train, min_nr_movies_test), "rb"))

#dictionary {user id 6 : movie id}
#cold_start = pickle.load(open("imperial_ijcai/test_%s_%s/cold_start_%s_%s.pkl" % (min_nr_movies_train, min_nr_movies_test, min_nr_movies_train, min_nr_movies_test), "rb"))

#{user id 6 : {directors: ..., sims: ..., actors: ..., genres: ...}}
#testing_users_cold_start = pickle.load(open("imperial_ijcai/test_%s_%s/testing_users_cold_start.pkl"  % (min_nr_movies_train, min_nr_movies_test), "rb"))


#rows = users, columns = genres
preferencesDF = pd.DataFrame.from_dict(preferences, orient = 'index')
preferencesDF.columns = preferencesDF.columns + '_genre'

#rows = users, columns = actors
actorsDF = pd.DataFrame.from_dict(all_actors, orient = 'index')
actorsDF.columns = actorsDF.columns + '_actors'

#rows = users, columns = directors
directorsDF = pd.DataFrame.from_dict(all_directors, orient = 'index')
directorsDF.columns = directorsDF.columns + '_director'

#rows = users, columns = users
similaritiesDF = pd.DataFrame.from_dict(all_similarities, orient = 'index')

class Recommend:

	def __init__(self, data, userAspectDF, itemAspectDF,similaritiesDF,aspectDict,typeList,itemIDCol, userIDCol, ratingCol):
		self.data = data

		self.userAspectDF = userAspectDF
		self.itemAspectDF = itemAspectDF
		self.aspectDict = aspectDict

		self.typeList = typeList
		self.itemIDCol = itemIDCol
		self.userIDCol = userIDCol
		self.ratingCol = ratingCol	

		self.itemList = data[self.itemIDCol].unique()
		self.userList = data[self.userIDCol].unique()

		self.typeDF = pd.DataFrame(0.0, index = self.userList,columns = self.typeList)

		self.similaritiesDF = similaritiesDF#pd.DataFrame({
		# 						'364518': {'2473170': 0.5, '1601783':0.1, '765331':0.9, '1476213':0.4, '364518':1},
		# 						'2473170': {'364518':0.5,'1601783': 0.2, '765331':0.5, '1476213' : 0.2,'2473170' :1},
		# 						'1601783' : {'2473170' :0.2, '364518' : 0.1, '765331': 0.7, '1476213': .15,'1601783' :1},
		# 						'765331' : {'364518' : 0.9, '2473170' : 0.5, '1601783' :0.7, '1476213' : .3,'765331':1},
		# 						'1476213' : {'364518' : 0.4,'2473170' : 0.2, '1601783' : .15 , '765331' :.3,'1476213':1}
		# 						})#pd.DataFrame.from_dict(all_similarities, orient = 'index')
		#self.muCF = pd.Series(np.random.rand(len(self.userList)),index = self.userList)
		self.muCF = pd.Series(0.3,index = self.userList)

	#add train data/full data argument
	def weightedAvgRatings(self):

		weighAvgDF = pd.DataFrame(0.0, index = self.userList,columns = self.itemList)

		for user_id in self.userList:

			for item_id in self.itemList:

				d = self.data[(self.data[self.itemIDCol] == item_id) & (self.data[self.userIDCol] != user_id)]

				numUsers = d.shape[0]

				simUsers = self.similaritiesDF.loc[user_id, d[self.userIDCol].values].values

				numerator = np.dot(d[self.ratingCol].values, simUsers)

				if numUsers > 0 and simUsers.sum() > 0:

					weighAvgDF.loc[user_id,item_id] = numerator / numUsers

		return weighAvgDF

	#add train data/full data argument
	def predAspectRatings(self):

		weighAvgDF = self.weightedAvgRatings()

		predAspectMat = pd.DataFrame(0.0, index = self.userList,columns = self.userAspectDF.columns)

		for user_id in self.userList:

			for aspect in self.userAspectDF.columns:

				#get movie ids rated by user u
				ids = self.data[self.itemIDCol][data[self.userIDCol] == user_id].values

				#item aspect ratings from user i 
				rated = self.itemAspectDF.loc[ids, aspect]

				lmbdaU = list(rated[rated > 0.0].index)


				##############################
				#linked items with defined weighted averages
				linkedItems = self.itemAspectDF[self.itemAspectDF[aspect] > 0.0].index

				#get movie ids in linkedItems that aren't in lmbdaU
				lmbdaInv = list(set(linkedItems) - (set(lmbdaU)))

				sumLmbdaInv = self.muCF[user_id] * weighAvgDF[lmbdaInv].values.sum()

				##############################

				len_lmbda = len(lmbdaU)
				len_lmbda_inv = len(lmbdaInv)

				sumRatings = self.data[self.data[self.itemIDCol].isin(lmbdaU) & (self.data[self.userIDCol] == user_id)][self.ratingCol].values.sum()
				
				currentRating = self.userAspectDF.loc[user_id, aspect]

				if currentRating > 0.0:

					predAspectMat.loc[user_id, aspect] = currentRating

				else: 

					if not lmbdaU and not lmbdaInv:

						predAspectMat.loc[user_id, aspect] = 0.0

					else:

						if not lmbdaU:

							predAspectMat.loc[user_id, aspect] = (sumLmbdaInv / len_lmbda) / (1 + self.muCF[user_id])

						else:

							if not lmbdaInv:

								predAspectMat.loc[user_id, aspect] = sumRatings / len_lmbda
								
							else:

								predAspectMat.loc[user_id, aspect] = ((sumRatings / len_lmbda) + (sumLmbdaInv / len_lmbda_inv )) / (1 + self.muCF[user_id])

		return weighAvgDF, predAspectMat

	def predItemRatings(self):

		'''Computes predicted item ratings for all users (training and test) for all items'''

		weighAvgDF, predAspectMat = self.predAspectRatings()

		#add test users to index list, add test items to itemlist
		predItemDF = pd.DataFrame(0.0, index = self.userList,columns = self.itemList)

		for user_id in self.userList:

			for item_id in self.itemList:

				ratingUserI = self.data[(self.data[self.itemIDCol] == item_id) & (self.data[self.userIDCol] == user_id)][self.ratingCol].values

				#check if rating is defined: list must be nonempty and element must be non zero
				if ratingUserI and 0 not in ratingUserI:

					predItemDF.loc[user_id, item_id] = ratingUserI[0]

				else:

					weightedRating = weighAvgDF.loc[user_id, item_id]

					typeSum = self.typeDF.loc[user_id,:].sum()

					if weightedRating and not typeSum:

						predItemDF.loc[user_id, item_id] = weightedRating

					else:

						#aspects = np.hstack(aspectDict[item_id].values())
						aspects = ['Action_genre', 'Adventure_genre', 'Mike Meyers_actors', 'Éva Gárdos_director']

						#count number of type occurances in aspects list
						aspectCount = {types : sum(types in aspect for aspect in aspects) for types in self.typeDF.columns}

						denom = self.typeDF.loc[user_id].sum()

						output = []

						#list of tuples (type, weight)
						zipped_types = list(zip(self.typeDF.loc[user_id].index,self.typeDF.loc[user_id]))

						#list of tuples (aspect, aspect rating)
						zipped_aspects = list(zip((predAspectsmatrix.loc[user_id,aspects]).index, predAspectsmatrix.loc[user_id,aspects]))

						output = sum([val* (p / aspectCount[t]) for t, val in zipped_types for a, p in zipped_aspects if a.endswith("_"+t)])

						if not weightedRating and typeSum > 0:

							predItemDF.loc[user_id, item_id] = output / denom

						else:

							if weightedRating and (self.muCF[user_id] + typeSum > 0):

								predItemDF.loc[user_id, item_id] = (self.muCF[user_id] + output) / (self.muCF[user_id] + denom)

							else:

								predItemDF.loc[user_id, item_id] = 0

		return predItemDF

	def predict(self):

		pass


userID = ['364518', '2473170', '1601783', '765331', '1476213', '364518', '364518','364518','364518', '1476213']

rating = [2,5,2,1,2,3,4,5,2,3]

itemID = ['tt0264150', 'tt0264150','tt0264150','tt0264149','tt0264149', 'tt0124718', 'tt0134618','tt0146838', 'tt0298148','tt0295178']

data = pd.DataFrame({'userID': userID, 'rating': rating, 'itemID' : itemID})

userAspectDF = pd.concat([actorsDF, directorsDF,preferencesDF], axis=1)


itemAspectDF = pd.DataFrame({'tt0264150': {'Action_genre': 1, 'Adventure_genre':1, 'Zach Braff_actors':0, 'Mike Meyers_actors':1, 'Éva Gárdos_director':1},
					'tt0264149': {'Action_genre': 0, 'Adventure_genre':1, 'Zach Braff_actors':1, 'Mike Meyers_actors':0, 'Éva Gárdos_director':1},
					'tt0124718' : {'Action_genre': 0, 'Adventure_genre':0, 'Zach Braff_actors':1, 'Mike Meyers_actors':1, 'Éva Gárdos_director':1},
					'tt0134618' : {'Action_genre': 1, 'Adventure_genre':1, 'Zach Braff_actors':0, 'Mike Meyers_actors':1, 'Éva Gárdos_director':1},
					'tt0146838' : {'Action_genre': 1, 'Adventure_genre':0, 'Zach Braff_actors':0, 'Mike Meyers_actors':0, 'Éva Gárdos_director':1},
					'tt0210945' : {'Action_genre': 1, 'Adventure_genre':1, 'Zach Braff_actors':1, 'Mike Meyers_actors':0, 'Éva Gárdos_director':1},
					'tt0245562' : {'Action_genre': 0, 'Adventure_genre': 1, 'Zach Braff_actors':1, 'Mike Meyers_actors':0, 'Éva Gárdos_director':1},
					'tt0349710' : {'Action_genre': 0, 'Adventure_genre':0, 'Zach Braff_actors':0, 'Mike Meyers_actors':1, 'Éva Gárdos_director':1},
					'tt0120630' : {'Action_genre': 1, 'Adventure_genre':0, 'Zach Braff_actors': 0, 'Mike Meyers_actors':1, 'Éva Gárdos_director':1},
					'tt0290673' : {'Action_genre': 0, 'Adventure_genre':1, 'Zach Braff_actors':1, 'Mike Meyers_actors':0, 'Éva Gárdos_director':1}
  								}).T

userAspectDF = pd.DataFrame({'364518': {'Action_genre': 3, 'Adventure_genre':3.5, 'Zach Braff_actors':5, 'Mike Meyers_actors':1, 'Éva Gárdos_director':4},
					'2473170': {'Action_genre': 2.5, 'Adventure_genre':3.5, 'Zach Braff_actors':1, 'Mike Meyers_actors':1.5, 'Éva Gárdos_director':3},
					'1601783' : {'Action_genre': 4, 'Adventure_genre':2, 'Zach Braff_actors':2, 'Mike Meyers_actors':3, 'Éva Gárdos_director':2},
					'765331' : {'Action_genre': 5, 'Adventure_genre':1, 'Zach Braff_actors':3, 'Mike Meyers_actors':3.5, 'Éva Gárdos_director':2.5},
					'1476213' : {'Action_genre': 1, 'Adventure_genre':3, 'Zach Braff_actors':4.5, 'Mike Meyers_actors':3.5, 'Éva Gárdos_director':3},
					'90560' : {'Action_genre': 0, 'Adventure_genre':1, 'Zach Braff_actors':0, 'Mike Meyers_actors':3, 'Éva Gárdos_director':0},
					'1499504' : {'Action_genre': 0, 'Adventure_genre':4, 'Zach Braff_actors':0, 'Mike Meyers_actors':0, 'Éva Gárdos_director':5}
								}).T

testuser = ['90560','90560','90560','90560','1499504']

ratingtest = [4,3,3,1,1]

itemidtest = ['tt0210945', 'tt0245562', 'tt0349710', 'tt0120630','tt0290673']

testDF = pd.DataFrame({'userID': testuser, 'rating': ratingtest, 'itemID' : itemidtest})

_,_,testSimsDF = Loader.NetflixTest(10, 5, surprise=False)

similaritiesDF = similaritiesDF.append(testSimsDF)
similaritiesDF.fillna(0.0, inplace=True)	
trueLabels = testDF.pop('rating')


testDF['rating'] = 0.0

data = pd.concat([data, testDF], axis = 0, sort=True)

R = Recommend(data, userAspectDF,itemAspectDF,similaritiesDF,films,
				typeList = ['genre', 'actors','director'],
				itemIDCol = 'itemID', 
				userIDCol = 'userID', 
				ratingCol = 'rating')
R.typeDF['genre'] = 0.3
R.typeDF['actors'] = 0.5
R.typeDF['director'] = 0.2

weighAvgDF, predAspectsmatrix = R.predAspectRatings()

print(R.predItemRatings())
#pd.set_option('display.max_columns', None)  


#TO DO:
#Predict on test set
#Accuracy function


