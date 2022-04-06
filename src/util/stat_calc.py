#!/usr/bin/env python

"""This file contains methods to calculate statistics from dataframes.

	Description.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
	
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
	
	# Dropout Length
	if("dropout_length" in df.columns):
		df['dropout_length'] = df['lastcontact'].diff()[1:]
	else:
		df.insert(df.shape[1], 'dropout_length', df['lastcontact'].diff()[1:])
	
	# Mean
	mean = df['dropout_length'].mean()
	if("mean" in df.columns):
		df['mean'] = mean
	else:
		df.insert(df.shape[1], 'mean', mean)
		
	#print(df)
	return


def stdev_zscore(df):
	
	# Standard Deviation
	stdev = df['dropout_length'].std()
	if("stdev" in df.columns):
		df['stdev'] = stdev
	else:
		df.insert(df.shape[1], 'stdev', stdev)
	
	# Standard Deviation Z-Score
	stdev_zscores = stats.zscore(df['dropout_length'].dropna())
	if("stdev_zscore" in df.columns):
		df['stdev_zscore'] = stdev_zscores
	else:
		df.insert(df.shape[1], 'stdev_zscore', stdev_zscores)
	
	#print(df[['dropout_length', 'stdev', 'stdev_zscore']])
	return
	

def simple_moving_average(df, window):
	"""_summary_

	Args:
		df (_type_): _description_
		window (int): Size of the moving window
	"""
	
	# Create column name with window value included
	#window_colname = "window_size"
	
	# Window Size
	if("sma_window_size" in df.columns):
		df['sma_window_size'] = str(window)
	else:
		df.insert(df.shape[1], 'sma_window_size', str(window))
	
	
	# Simple Moving Average
	sma = df['dropout_length'].rolling(window).mean()
	if("sma" in df.columns):
		df['sma'] = sma
	else:
		df.insert(df.shape[1], 'sma', sma)
	
	#print(df[['dropout_length', colname]])
	""" import matplotlib.pyplot as plt
	df['dropout_length'].plot()
	df[colname].plot()
	plt.legend()
	plt.show() """
	return
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
	
	# Set column to numpy array
	arr = df['dropout_length']
	arr = np.asanyarray(arr)
	
	# Mean and standard deviation
	mean = np.nanmean(arr, axis = axis)
	stdev = np.nanstd(arr, axis = axis, ddof = ddof)
	
	# Signal to Noise Ratio
	snr = np.where(stdev == 0, 0, mean / stdev)
	
	#print(snr)
	return snr
	
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
	
	
def snr_rolling(df, axis, ddof):
	
	# Initialize list
	snr_list = []
	
	# Loop through list calculating and appending Signal to Noise Ratio
	for i in range(df.shape[0]):
		snr_list.append(signal_noise_ratio(df[:i], axis, ddof))
	
	# Signal to Noise Ratio (Rolling)
	if("snr_rolling" in df.columns):
		df['snr_rolling'] = snr_list
	else:
		df.insert(df.shape[1], 'snr_rolling', snr_list)
	
	#print(df[['dropout_length', 'snr_rolling']])
	return
	

def mode_deviation(df):
	"""_summary_
	Mode Deviation Algorithm by Akshay Ramchandra, University of North Dakota, April 4 2022

	Args:
		df (_type_): _description_
	"""
	
	
	# Calculate 'true mode' by removing values <=0
	df_temp = df.copy()
	df_temp.loc[df_temp['dropout_length'] <= 0] = None
	true_mode = df_temp['dropout_length'].mode()[0]
	
	# Calculate mode
	mode = df['dropout_length'].mode()[0]
	
	# Use true mode?
	#print(mode, true_mode)
	use_true_mode = True
	if(use_true_mode):
		mode = true_mode
	
	# Mode 
	if("mode" in df.columns):
		df['mode'] = mode
	else:
		df.insert(df.shape[1], 'mode', mode)
	
	
	# Calculate Mode Deviation
	arr = df['dropout_length']
	sig = 0
	counter = 0
	for i in arr:
		if i > mode:
			counter += 1
			sig += ((i - mode) ** 2)
	if counter > 0:
		mode_dev = (sig / counter) ** 0.5
	else:
		mode_dev = 0
	
	# Mode Deviation
	#print("Mode Deviation: {}".format(mode_dev))
	if("mode_dev" in df.columns):
		df['mode_dev'] = mode_dev
	else:
		df.insert(df.shape[1], 'mode_dev', mode_dev)
	
	
	# Calculate Mode Deviation Z-Score
	mode_dev_zscores = []
	for i in arr:
		z = (i - mode) / mode_dev
		mode_dev_zscores.append(z)
	
	# Mode Deviation Z-Scores
	#print("Mode Deviation Z-Scores: {}".format(mode_dev_zscores))
	if("mode_dev_zscore" in df.columns):
		df['mode_dev_zscore'] = mode_dev_zscores
	else:
		df.insert(df.shape[1], 'mode_dev_zscore', mode_dev_zscores)
	
	#print(df[['dropout_length', 'mode_dev', 'mode_dev_zscore']])
	return


def score(df):
	
	# Calculate a score for each row
	scores = []
	for i, row in df.iterrows():
		score = 0
		#print(row['dropout_length'])
		
		
		# dropout_length > mean
		if (row['dropout_length'] > row['mean']):
			score += 1
			
		# dropout_length > mode
		if (row['dropout_length'] > row['mode']):
			score += 1
		
		# dropout_length > sma25
		if (row['dropout_length'] > row['sma']):
			score += 1
		
		# dropout_length > snr_rolling
		if (row['dropout_length'] > row['snr_rolling']):
			score += 1
		
		# stdev_zscore > x
		if (row['stdev_zscore'] > 1):
			score += 1
		if (row['stdev_zscore'] > 2):
			score += 1
		if (row['stdev_zscore'] > 3):
			score += 1
		
		# mode_dev_zscore > x
		if (row['mode_dev_zscore'] > 1):
			score += 1
		if (row['mode_dev_zscore'] > 2):
			score += 1
		if (row['mode_dev_zscore'] > 3):
			score += 1
		
		scores.append(score)
		
	# Dropout Score
	if("score" in df.columns):
		df['score'] = scores
	else:
		df.insert(df.shape[1], 'score', scores)
	
	fig, axs = plt.subplots(2, 1, figsize = (10, 8))
	axs[0].plot(df['time'], df['score'], label = "score value")
	axs[0].set_title("Score vs Time")
	axs[0].set_xlabel("time (s)")
	axs[0].set_ylabel("score")
	
	axs[1].plot(df['time'], df['dropout_length'], label = "dropout_length")
	axs[1].set_title("Dropout Length vs Time")
	axs[1].set_xlabel("time (s)")
	axs[1].set_ylabel("dropout_length (s)")
	
	plt.show()
	plt.clf()
	
	from matplotlib import rcParams
	rcParams['figure.figsize'] = 10, 8
	sns.scatterplot(data = df, x = "time", y = "score", hue = "score", palette=sns.dark_palette("#FF0000", as_cmap=True))
	plt.title("Score vs Time")
	plt.xlabel("time (s)")
	plt.ylabel("score value")
	plt.show()
	
	return df

def autotag(df):
	
	# Score Counts
	print(df['score'].value_counts().sort_index())
	
	# Generate tag for each row
	tags = []
	dropout_threshold = 4
	for i, row in df.iterrows():
		
		if(row['score'] <= dropout_threshold):
			df.at[i, 'taxonomy'] = "noise"
			
		if(row['score'] > dropout_threshold):
			df.at[i, 'taxonomy'] = "dropout"
		
		if(row['dropout_length'] <= row['mode']):
			df.at[i, 'taxonomy'] = "normal"
		
		if(row['dropout_length'] <= 0):
			df.at[i, 'taxonomy'] = "erroneous"
	
	hue_order = ['normal', 'erroneous', 'noise', 'dropout']
	sns.scatterplot(data = df, x = "time", y = "dropout_length", hue = "taxonomy", hue_order = hue_order)
	plt.title("Dropout Length vs Time (Colored by Label)")
	plt.xlabel("time (s)")
	plt.ylabel("dropout_length (s)")
	plt.show()
	
	# Taxonomy Counts
	print(df['taxonomy'].value_counts().sort_index())
	
	return df


























































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
	
