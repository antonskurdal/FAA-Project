#!/usr/bin/env python

"""Flight Stats Calculator

	Modifies CSVs or Parquets in-place by adding a variety of statistical columns
"""

from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
import time

__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2020, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"



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
	
	return df
	
	
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
	
	return df


def simple_moving_average(df, window):
	"""_summary_

	Args:
		df (_type_): _description_
		window (int): Size of the moving window
	"""
	
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
	
	return df


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
	
	return snr


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
	
	return df


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
	if("mode_dev_zscore" in df.columns):
		df['mode_dev_zscore'] = mode_dev_zscores
	else:
		df.insert(df.shape[1], 'mode_dev_zscore', mode_dev_zscores)
	
	return df


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
	
	# Taxonomy Counts
	#print(df['taxonomy'].value_counts().sort_index())
	
	return df



# Set up directory
parent_directory = Path("D:/#FAA UAS Project/OpenSky WEEK/Individual Aircraft/batch/output")

file = "a052b6_.parquet"
#file = "a47dac_.parquet"

data = pd.read_parquet(parent_directory / file)
print(data.head())

exit()


# Iterate through directory and count files
extensions = ('*.csv', '*.parquet')
file_count = 0
for ext in extensions:
	for file in parent_directory.rglob(ext):
		print(file.name)
		file_count += 1


# Loop through files and calculate flight stats
count = 1
for ext in extensions:
	for file in parent_directory.rglob(ext):
		
		# Display count
		print("({}/{}) Processing '{}'".format(count, file_count, file.name))
		count += 1
		
		# Initialize variables
		data = pd.DataFrame()
		splits = []
		threshold = 900
		
		# Load file
		if(file.suffix == '.csv'):
			data = pd.read_csv(file)
		elif(file.suffix == '.parquet'):
			data = pd.read_parquet(file)
		else:
			print("Invalid file extension.")
		
		# Calculate Statistics
		data = dropouts(data)
		data = stdev_zscore(data)
		data = simple_moving_average(data, 10)	# Simple Moving Average (window = 10)
		data = snr_rolling(data, 0, 0) # Signal to Noise Ratio (Rolling)
		data = mode_deviation(data) # Mode Deviation Z-Score
		
		# Calculate score & autotag
		data = score(data)
		data = autotag(data)
		
		# Save file
		data.to_csv(file)