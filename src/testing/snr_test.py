from pathlib import Path
from statistics import median
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



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
	m = np.nanmean(a, axis = axis)
	
	
	print("mean: {}".format(m))
	
	sd = np.nanstd(a, axis = axis, ddof = ddof)
	
	print("mean: {}".format(m))
	
	
	snr = np.where(sd == 0, 0, m / sd)
	
	print("snr:\n{}".format(snr))
	
	#plt.plot(snr)
	#plt.show()
	
	
	return np.where(sd == 0, 0, m / sd)
	

def reg(df):
	
	
	reg = linear_model.LinearRegression()
	
	#mode_line = np.full(shape = df[['dropout_length']].dropna().shape[0], fill_value=df['dropout_length'].mode()[0] )
	print(df['dropout_length'].shape[0])
	
	df.loc[df['dropout_length'] <= 0] = None
	mode = df['dropout_length'].mode()[0]
	
	dfshape = df['dropout_length'].dropna().shape[0]
	print("dfshape: {}".format(dfshape))
	mode_line = np.full(shape = dfshape, fill_value = mode)
	
	print(mode)
	print(len(mode_line))
	
	x = df['dropout_length'].dropna().to_numpy().reshape(-1,1)
	
	#print("nplen: {}".format(len(df['dropout_length'].dropna().to_numpy().reshape(-1,1))))
	print("nplen: {}".format(len(x)))
	
	reg = reg.fit(x, mode_line)
	
	#print(reg.coef_)
	
	plt.scatter(x.index, x)
	
	#plt.plot(mode_line, reg.predict(df[['dropout_length']]))
	plt.show()



def reg_rmse(df):
	# Sklearn regression metric
	#y = val > mode 
	#y^ is mode
	from sklearn.metrics import mean_squared_error
	
	df.loc[df['dropout_length'] <= 0] = None
	
	true_mode = df['dropout_length'].mode()[0]
	mode_filtered = df['dropout_length'].loc[df['dropout_length'] >= true_mode]
	
	# y = regression line
	# y^ = new y values
	# error -> y - y^
	
	y_true = mode_filtered
	y_pred = np.ones(mode_filtered.shape[0])*true_mode
	
	#y_true = np.ones(mode_filtered.shape[0])*true_mode
	#y_pred = mode_filtered
	
	
	print(mean_squared_error(y_true, y_pred, squared=True))
	
	plt.scatter(y_true-y_pred, y_pred)
	plt.show()
	




def mode_deviation(df):
	
	
	# Calculate true mode by remove values <= 0
	df.loc[df['dropout_length'] <= 0] = None
	mode = df['dropout_length'].mode()[0]
	print("Mode: {}".format(mode))
	
	arr = df['dropout_length']
	sig = 0
	
	# Find Mode Deviation
	counter = 0
	for i in arr:
		if i > mode:
			counter += 1
			sig += ((i - mode) ** 2)
	if counter > 0:
		mode_dev = (sig / counter) ** 0.5
		#return (sig / counter) ** 0.5
	else:
		mode_dev = 0
		#return 0
	
	print("Mode Deviation: {}".format(mode_dev))
	
	# Find Mode Z-Score
	mode_zscores = []
	counter = 0
	for i in arr:
		z = (i - mode) / mode_dev
		mode_zscores.append(z)
		print("(i, z): ({}, {})".format(i, z))
	
	print(mode_zscores)

	
	# import seaborn as sns
	# sns.displot(mode_zscores, kind = 'kde')
	# sns.displot(df['dropout_zscore'], kind = 'kde')
	
	# Plot Std and Mode Deviation
	pd.Series(mode_zscores).plot.kde(label = "mode dev zscores")
	df['dropout_zscore'].plot.kde(label = "std dev zscores")
	plt.legend()
	plt.title("STD vs MODE Deviation")
	plt.show()
	return mode_dev, mode_zscores





def snr_rolling(df):
	
	snr_list = []
	
	for i in range(df.shape[0]):
		print(i)
		snr_list.append(signal_noise_ratio(df[:i], 0, 0))
	
	#print(snr_list)
	
	df2 = df.copy()
	
	mode_dev, mode_zscores = mode_deviation(df2)
	
	
	
	plt.plot(df['dropout_length'], label = "dropout_length")
	plt.plot(snr_list, label = "rolling snr")
	plt.plot(df['dropout_sma25'], label = "sma25")
	plt.plot(df['dropout_zscore'], label = "dropout_zscore")
	
	df.loc[df['dropout_length'] <= 0] = None
	#print(df)
	
	mode = df['dropout_length'].mode()[0]
	
	mean = df['dropout_length'].mean()
	
	std = df['dropout_length'].std()
	
	#print(df.where(df['dropout_length'] > 0))
	
	#print(df['dropout_length'].mode()[0])
	print(mode)
	
	
	
	
	
	plt.plot(mode_zscores, label = "mode zscore")
	
	
	
	
	
	plt.axhline(mode, linestyle = "--", color = "black", label = "mode")
	plt.axhline(mean, linestyle = "--", color = "gray", label = "mean")
	plt.axhline(std, linestyle = "--", color = "pink", label = "std")
	plt.axhline(std*2, linestyle = "--", color = "magenta", label = "std*2")
	plt.axhline(std*3, linestyle = "--", color = "purple", label = "std*3")
	import matplotlib.patheffects as pe
	plt.axhline(mode_dev, lw = 2, linestyle = "--", color = "white", path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()], label = "mode dev")
	
	
	
	plt.legend()
	
	
	plt.xlabel("time/index")
	plt.ylabel("value")
	plt.show()



file = Path(Path.cwd() / "data" / "a8c313_modified.csv")

data = pd.read_csv(file)














#reg(data)
#reg_rmse(data)
#exit(0)


#print(data.columns)
#signal_noise_ratio(data, 0, 0)

snr_rolling(data)


#print(mode_deviation(data))