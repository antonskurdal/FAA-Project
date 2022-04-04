#!/usr/bin/env python

"""This file contains methods to calculate statistics from dataframes.

	Description.
"""

import pandas as pd
import numpy as np

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



def dropouts(df):
	
	if("dropout_length" in df.columns):
		df['dropout_length'] = df['lastcontact'].diff()[1:]
	else:
		df.insert(df.shape[1], 'dropout_length', df['lastcontact'].diff()[1:])
	
	print(df)


def zscore(df):
	import scipy.stats as stats
	
	
	zscores = stats.zscore(list(df['dropout_length'].dropna()))
	zscores = np.insert(zscores, 0, np.NaN, axis = 0)
		
	if("dropout_zscore" in df.columns):
		df['dropout_zscore'] = zscores
	else:
		df.insert(df.shape[1], 'dropout_zscore', zscores)
	
	
	print("[zscore]:\n")
	print(df[["dropout_length", "dropout_zscore", ]])
	

def simple_moving_average(df, window):
	"""_summary_

	Args:
		df (_type_): _description_
		window (int): Size of the moving window
	"""
	colname = "dropout_sma" + str(window)
	
	if(colname in df.columns):
		print(list(df['dropout_length'].rolling(window).mean()))
		df[colname] = list(df['dropout_length'].rolling(window).mean())
	else:
		df.insert(df.shape[1], colname, list(df['dropout_length'].rolling(window).mean()))
	
	print("[sma{}]:\n".format(window))
	print(list(df['dropout_length'].rolling(window).mean()))
	#y = list(df['dropout_length'].rolling(window))
	
	
	
	#df['dropout_length'].plot()
	
	"""
	import matplotlib.pyplot as plt
	df.reset_index()
	df['dropout_length'].plot()
	df['dropout_length'].rolling(window).mean().plot()
	plt.legend()
	plt.show()
	"""
	
	print(df[["dropout_length", colname]])

	


def signal_noise_ratio(df, axis, ddof):
	"""_summary_
	Source: https://www.geeksforgeeks.org/scipy-stats-signaltonoise-function-python/
	Args:
		df (_type_): [array_like] Input dataframe to be converted into array or object having the elements to calculate the signal-to-noise ratio
		axis (_type_): Axis along which the mean is to be computed. By default axis = 0.
		ddof (_type_): Degree of freedom correction for Standard Deviation.

	Returns:
		snr (array): mean to standard deviation ratio i.e. signal-to-noise ratio.
	"""
	
	
	
	a = df['dropout_length']
	
	a = np.asanyarray(a)
	m = a.mean(axis)
	sd = a.std(axis = axis, ddof = ddof)
	
	
	snr = np.where(sd == 0, 0, m / sd)
	
	import matplotlib.pyplot as plt
	
	print("snr:\n{}".format(snr))
	
	plt.plot(snr)
	plt.show()
	
	
	return np.where(sd == 0, 0, m / sd)
	
	





def linear_regression(df):
	
	import numpy as np
	import matplotlib.pyplot as plt  # To visualize
	import pandas as pd  # To read data
	from sklearn.linear_model import LinearRegression
	
	X = df['dropout_length'].values.reshape(-1, 1)  # values converts it into a numpy array
	print("[regression] X: {}".format(list(X)))
	
	Y = df.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
	print("[regression] Y: {}".format(list(Y)))
	
	# linear_regressor = LinearRegression()  # create object for the class
	# linear_regressor.fit(X, Y)  # perform linear regression
	# Y_pred = linear_regressor.predict(X)  # make predictions
	
	# plt.scatter(X, Y)
	# plt.plot(X, Y_pred, color='red')
	# plt.show()
	
