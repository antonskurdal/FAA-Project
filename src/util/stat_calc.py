#!/usr/bin/env python

"""This file contains methods to calculate statistics from dataframes.

	Description.
"""

import tkinter as tk
from tkinter import *
from pathlib import Path
from tkinter import messagebox
import pandas as pd
from dataclasses import dataclass
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

import util.sku_widgets as sku
import util.grapher as grapher

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







def set_test(x):
	x = "Goodbye"
	return x
	
def apply_taxonomy(df, col, bounds, type):
	
	# print(df)
	# #x = df.xs_colname
	# #print(x)
	# print(bounds)
	
	# df['taxonomy'] = type
	
	# return df
	
	# Get data lower than bounds
	low = df[df[col] < bounds[0]]
	
	# Get data higher than bounds
	high = df[df[col] > bounds[1]]
	
	mid = df[(df[col] > bounds[0]) & (df[col] < bounds[1])]
	#print(mid)
	
	mid['taxonomy'] = type
	
	df = pd.concat([low, mid, high])
	print("\n[STAT_CALC]:")
	print(df)
	print("\n")
	
	return df
	
	""" x = self.obj.xs_colname
	y = self.obj.ys_colname
	data = self.obj.current
	
	# Get data lower than bounds
	low = data[data[x] < bounds[0]]
	
	# Get data higher than bounds
	high = data[data[x] > bounds[1]]
	
	# Remove lower than bounds
	mid = data[data[x] >= bounds[0]]
	
	# Remove data higher than bounds
	mid = mid[mid[x] <= bounds[1]]
	
	mid = mid.reset_index(drop = True)
	
	# Modify
	for i in range(len(mid[y])):
		
		rand = randint(percent*-1, percent)
		
		if (rand != 0):
			rand_pct = rand/100
		else:
			rand_pct = 0
		
		mid.at[i, y] = mid.at[i, y] + (mid.at[i, y] * rand_pct)
	
	
	# Concat data frames
	data = pd.concat([low, mid, high])
	data = data.reset_index(drop = True)
	
	self.obj.current = data """