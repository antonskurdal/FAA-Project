#!/usr/bin/env python

"""This file contains methods to calculate statistics from dataframes.

	Description.
"""

import pandas as pd

__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2020, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"


# Make sure code runs as a module
if(__name__ == '__main__'):
	print("This code is meant to be run as a module.")
	exit(0)
	


"""ADD FUNCTION TO APPLY ZSCORE, TIME SINCE LAST CONTACT, ETC TO DATAFRAME"""

	
def apply_taxonomy(df, col, bounds, type):
	
	# Get data lower than bounds
	low = df[df[col] < bounds[0]]
	
	# Get data higher than bounds
	high = df[df[col] > bounds[1]]
	
	mid = df[(df[col] > bounds[0]) & (df[col] < bounds[1])]
	
	mid['taxonomy'] = type
	
	df = pd.concat([low, mid, high])
	print("\n[STAT_CALC]:")
	print(df)
	print("\n")
	
	return df



def get_dropouts(df):
	
	df.insert(df.shape[1], 'dropout_length', df['lastcontact'].diff()[1:])
	print(df)
	
	
def regression(df):
	
	import numpy as np
	import matplotlib.pyplot as plt  # To visualize
	import pandas as pd  # To read data
	from sklearn.linear_model import LinearRegression
	
	X = df['dropout_length'].values.reshape(-1, 1)  # values converts it into a numpy array
	print("[regression] X: {}".format(X))
	
	Y = df.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
	print("[regression] Y: {}".format(Y))
	
	# linear_regressor = LinearRegression()  # create object for the class
	# linear_regressor.fit(X, Y)  # perform linear regression
	# Y_pred = linear_regressor.predict(X)  # make predictions
	
	# plt.scatter(X, Y)
	# plt.plot(X, Y_pred, color='red')
	# plt.show()