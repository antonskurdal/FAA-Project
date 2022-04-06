from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
		if (row['dropout_length'] > row['sma25']):
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
	axs[0].plot(df['score'])
	axs[1].plot(df['dropout_length'])
	plt.show()
	plt.clf()
	
	from matplotlib import rcParams
	rcParams['figure.figsize'] = 10, 8
	sns.scatterplot(data = df, x = df.index, y = "score", hue = "score", palette=sns.dark_palette("#FF0000", as_cmap=True))
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
	
	sns.scatterplot(data = df, x = df.index, y = "dropout_length", hue = "taxonomy")
	plt.show()
	
	# Taxonomy Counts
	print(df['taxonomy'].value_counts().sort_index())
	
	return df
		
	


file = Path(Path.cwd() / "data" / "autotag tests" / "a0f7db_autotag_base.csv")
data = pd.read_csv(file)

data_scored = score(data)

data_tagged = autotag(data_scored)


print(file.parent)
data_tagged.to_csv(Path(file.parent / Path(str(file.stem) + "_labeled.csv")))