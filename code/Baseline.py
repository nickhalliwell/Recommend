#!/usr/bin/env python3

#import os
#os.chdir('/Users/nhalliwe/Desktop/code')

import netflixLoaderBranch as Loader
from surprise import CoClustering, KNNBasic, KNNWithZScore,NMF, SlopeOne, SVD
from surprise import Reader
from surprise import Dataset
import pandas as pd
import numpy as np
import sys

helper_path = os.path.abspath('helper.py')
sys.path.insert(0, helper_path)

import helper

np.random.seed(12345)

def getData(min_nr_movies_train,min_nr_movies_test):

	cols = ['userID', 'itemID', 'rating']

	reader = Reader(rating_scale=(1, 5))

	if netflix=True:

		train, test = Loader.Netflix(min_nr_movies_train,min_nr_movies_test, surprise=True)

		trainData = pd.DataFrame(train)

		testData = pd.DataFrame(test)


		trainset =  Dataset.load_from_df(trainData[cols], reader).build_full_trainset()

		testset = Dataset.load_from_df(testData[cols], reader).build_full_trainset()

		testset = testset.build_testset()

	else:

		amazon = helper.getDF('/Amazon-data/reviews_Books.json.gz')



	return trainset, testset

def run(algorithm,min_nr_movies_train, min_nr_movies_test, netflix=True):

	if netflix:

		trainset, testset = getNetflix()

	else:



	if algorithm == 'SVD':

		model = SVD()

	elif algorithm == 'KNNBasic':

		model = KNNBasic()

	elif algorithm == 'KNNWithZScore':

		model = KNNWithZScore()

	elif algorithm == 'NMF':

		model = NMF()

	elif algorithm == 'CoClustering':

		model = CoClustering()

	elif algorithm == 'SlopeOne':

		model = SlopeOne()

	model.fit(trainset)
	preds = model.test(testset)

	predsDF = pd.DataFrame(preds)
	predsDF['userID'] = predsDF['uid']
	predsDF['itemID'] = predsDF['iid']
	predsDF['pred'] = predsDF['est']

	full = pd.merge(predsDF, testData, on = ['userID','itemID'], how = 'inner')

	out = []
	for pred, rating in zip(round(full['pred']), full['rating']):
	    
	    if pred==rating:
	    #if (pred + 1 == rating) or (pred - 1 == rating) or (pred == rating):
	        
	        out.append(1)
	    else:
	        out.append(0)

	acc = sum(out) / len(out)

	return 'Model:{}, Accuracy: {}, Num Movies: {}/{}'.format(algorithm, round(acc,3) * 100, min_nr_movies_train, min_nr_movies_test)

if __name__ == "__main__":

	print(run('CoClustering',10, 5))
	print(run('CoClustering',20, 5))
	print(run('CoClustering',20, 7))
	print(run('CoClustering',20, 10))	

	print(run('KNNBasic',10, 5))
	print(run('KNNBasic',20, 5))
	print(run('KNNBasic',20, 7))
	print(run('KNNBasic',20, 10))

	print(run('KNNWithZScore',10, 5))
	print(run('KNNWithZScore',20, 5))
	print(run('KNNWithZScore',20, 7))
	print(run('KNNWithZScore',20, 10))	

	print(run('NMF',10, 5))
	print(run('NMF',20, 5))
	print(run('NMF',20, 7))
	print(run('NMF',20, 10))	

	print(run('SlopeOne',10, 5))
	print(run('SlopeOne',20, 5))
	print(run('SlopeOne',20, 7))
	print(run('SlopeOne',20, 10))
	
	print(run('SVD',10, 5))
	print(run('SVD',20, 5))
	print(run('SVD',20, 7))
	print(run('SVD',20, 10))


