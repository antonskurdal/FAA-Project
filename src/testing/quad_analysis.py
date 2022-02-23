import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
import seaborn as sns
import sys

pd.set_option('display.max_columns', None)


def calc_dropouts(directory, outfile, focus_col):
	"""Calculates the difference between the current column value and the previous for the specified column

	Args:
		infile (WindowsPath): Location of the input file
		outfile (WindowsPath): Location of the output file
		focus_col (string): column to analyze
	"""
	
	
	#Get list of file paths
	#pathslist = directory.glob('**/*.csv')
	""" 	pathslist = []
	for filename in sorted(directory.glob('**/*.csv'), key=lambda path: int(path.name[:-4])):
		pathslist.append(filename)
		#print(filename) """
	
	pathslist = sorted(directory.glob('**/*.csv'), key=lambda path: int(path.name[:-4]))
	print(type(pathslist))
	
	
	#Initialize counter and loop through paths
	counter = 1
	file_count = len(list(directory.glob('**/*.csv')))
	for i, path in enumerate(pathslist):
		print("Processing ({}/{}) '{}'".format(i+1, file_count, path.name))
		
		df = pd.read_csv(path)
		
		#Dropout length
		df.insert(df.shape[1], 'dropout_length', df[focus_col].diff()[1:])
		
		#Mean
		df.insert(df.shape[1], 'avg', df['dropout_length'][1:].mean())

		#Mode
		df.insert(df.shape[1], 'mode', df['dropout_length'][1:].mode())
		df['mode'] = df['mode'][0]
		
		#Standard Deviation
		df.insert(df.shape[1], 'stddev', df['dropout_length'][1:].std())

		#Z-Score
		zscores = stats.zscore(list(df['dropout_length'].dropna()))
		zscores = np.insert(zscores, 0, np.NaN, axis = 0)
		df.insert(df.shape[1], 'zscore', zscores)
		
		#Create distribution plot
		g = sns.displot(df, x = "zscore", kind = "kde", fill = True)
		#ax.set(xlim=(-3.5,3.5))
		
		
		#Display figure
		plt.tight_layout()
		#plt.show()
		fig = g.fig
		pdf.savefig(fig)
		plt.close("all")	
	
		#print(df)
	
	

directory = Path(Path.cwd() / "data/DeliveryQuadcopterDroneData/individual_flights")
infile = Path(Path.cwd() / "data/DeliveryQuadcopterDroneData" / "flights.csv")
outfile = Path(Path.cwd() / "output/DeliveryQuadcopter" / "individual_flights.pdf")
#Prepare data and files
pdf = PdfPages(outfile)


calc_dropouts(directory, outfile, "time")
#calc_dropouts(Path(Path.cwd() / "data/DeliveryQuadcopterDroneData/individual_flights" / "2.csv"), outfile, "time")

#Close PDF
pdf.close()