#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(3)
num_users = 1
num_types = 1
num_aspects = 1
num_items = 1

#######################################################
#Read full data

X_train = None
A_train = None
R_train = None

X_test = None
A_test = None
R_test = None
#######################################################


#placeholders
#X matrix of size #items x #aspects
X = tf.placeholder(tf.float32, shape = [num_items, num_aspects], name = 'X')

#A matrix of size #users x #aspects
A = tf.placeholder(tf.float32, shape=[None, num_aspects], name="A")

#R read target value matrix of dimensions #users x #items
R = tf.placeholder(tf.float32, shape = [None, num_items], name = 'R')
#######################################################


#######################################################
#compute W_uv using A
#W_uv is matrix of size #users x #users
W_uv = tf.convert_to_tensor(cosine_similarity(A), name = 'W_uv')
#######################################################


#######################################################
#MODEL PARAMETERS
#mu_cf is vector of size #users x 1
mu_cf = tf.Variable(tf.random_uniform(shape = [num_users], minval = 0, maxval=1), name = "mu_cf")

#MU_t is matrix of size #users x #types (categories)
MU_t = tf.Variable(tf.random_uniform(shape = [num_users, num_types], minval = 0, maxval=1), name = 'MU_t')
#######################################################


#######################################################
#Weighted Average Ratings
#matrix of size #users x #items
weightedAvgRatings = tf.tanh(tf.matmul(W_uv, R))
#######################################################

#######################################################
#Rredicted Average Ratings
#matrix of size #users x #aspects
predAvgRatings = tf.tanh(tf.multiply(tf.matmul(weightedAvgRatings, X),mu_cf))
#######################################################

#######################################################
#Predicted Item Ratings
#matrix of size #users x #items
squareMU = tf.matmul(MU_t, MU_t, transpose_b=True)

weighPredAvg = tf.matmul(tf.matmul(squareMU, predAvgRatings), X, transpose_b=True)

predItemRatings = tf.tanh(weighPredAvg)
#######################################################


#assert predItemRatings and R have same shape
#tf.assert(tf.equal(predItemRatings.shape, R.shape))
cost = tf.losses.mean_squared_error(labels = R, predictions = predItemRatings)

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()


with tf.Session() as sess:

	sess.run(init)

	train_epoch = 1

	for epoch in range(train_epoch):

		for (x, a, r) in zip(X_train, A_train, R_train):

			sess.run(optimizer, feed_dict = {X:x, A:a, R: r})



	print('Done Training!')

	train_cost = sess.run(cost, feed_dict = {X: X_train, A: A_train, R: R_train})

	print("Training cost: ", train_cost)

	print('Final shape of predItemRatings:', sess.run(predItemRatings.shape))

	#test_cost = tf.losses.mean_squared_error(labels = R, predictions = predItemRatings)



def test(testData, A_train, weightedAvgRatings, R_train):


	#append test users to A_train, to create FULL_A, compute cos sim
	#append zeros to R_train (to prevent using similar users in test set)
	#



