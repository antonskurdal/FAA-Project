#!/usr/bin/env python

"""Flight Separator

	Description.
"""

from pathlib import Path
import pandas as pd
import plotly.express as px

__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2020, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"



directory = Path.cwd() / "data" / "test" / "flight_sep"
file = "addf73.csv"
#file = "a5681f.csv"


df = pd.read_csv(directory / file)

print(df.head())
print(df['icao24'].unique())

#df = df.drop_duplicates(subset = ["time", "lat", "lon", "geoaltitude"], keep = "last")
#df = df[:6517]
df['time'] = pd.to_datetime(df['time'],unit='s')

fig = px.scatter(df, x = df.index, y = "time", title = "[icao24: {}] Time by Index".format(df['icao24'].unique()[0]), hover_data={'index':(':.d', df.index), "time":':.d'})
fig.show()

df = df.sort_values(by=['time'])
df = df.reset_index(drop = True)

fig = px.scatter(df, x = df.index, y = "time", title = "[icao24: {}] Time by Index".format(df['icao24'].unique()[0]), hover_data={'index':(':.d', df.index), "time":':.d'})
fig.show()



fig = px.line(df, x = df.index, y = df['time'].diff())
fig.show()
exit()

fig = px.scatter_3d(
	df, 
	x = "lat", 
	y = "lon", 
	z = "geoaltitude", 
	color = df['time'], 
	color_continuous_scale=px.colors.sequential.Inferno,
	title = "[icao24: {}] Time by Index".format(df['icao24'].unique()[0])
)#, hover_data={'index':(':.d', df.index), "time":':.d'})
fig.show()



df.insert(0, 'time_diff', df['time'].diff())
print(df.head())
print(df.columns)
print(df['time_diff'].mode())

splits = []
mode = df['time_diff'].mode()[0]
for index, row in df.iterrows():
	#print(index, row['time_diff'])
	
	if(row['time_diff'] == None):
		pass
	elif(row['time_diff'] < 0):
		splits.append(index)
	elif(row['time_diff'] > mode):
		splits.append(index)

#print(splits)

df.to_csv("nodupes.csv")