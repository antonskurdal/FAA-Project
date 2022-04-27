#!/usr/bin/env python

"""Flight Separator

	Description.
"""

from pathlib import Path
from matplotlib.colors import Colormap
from matplotlib.pyplot import legend
import pandas as pd
import plotly.express as px
import time
import numpy as np

__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2020, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"


def clean_data(df):
	
	df = df.drop_duplicates(subset = ["time", "lat", "lon", "geoaltitude"], keep = "last")
	df = df.reset_index(drop = True)
	df.insert(0, 'time_diff', df['time'].diff())
	
	return df
	

def find_splits(df, threshold):
	
	splits = []
	mode = df['time_diff'].mode()[0]
	
	for index, row in df.iterrows():
		if(row['time_diff'] == None):
			pass
		elif(row['time_diff'] < 0):
			splits.append(index)
		# elif(row['time_diff'] > mode):
		# 	splits.append(index)
		elif(row['time_diff'] > threshold):
			splits.append(index)
	
	#print("Splits: {}".format(splits))
	
	return df, splits
	

def label_flights(df, splits):
	
	# Label splits as separate flights/trips
	prev_idx = 0
	
	for i, val in enumerate(splits):
		
		# Handle first value in splits
		if(i == 0):
			prev_idx = 0
		else:
			prev_idx = splits[i-1]
		
		# Label split range
		#print("[i = {}][prev_idx = {}][val = {}]".format(i, prev_idx, val))
		df.loc[prev_idx:val, 'flight_number'] = i
		prev_idx = val
		
		# Handle last value in splits
		if(i == len(splits)-1):
			prev_idx = val
			final_idx = df.shape[0]
			#print("[i = {}][prev_idx = {}][val = {}]".format(i+1, prev_idx, val))
			
			df.loc[prev_idx:final_idx, 'flight_number'] = i+1
		
	#print("Flights: {}".format(df['flight_number'].unique()))
	
	return df


# Set up directory
#parent_directory = Path("D:/#FAA UAS Project/OpenSky WEEK/Individual Aircraft/batch_3")
parent_directory = Path(Path.cwd() / "data" / "test" / "flight_sep" / "problems")
directory = parent_directory / "output/"
directory.mkdir(parents=True, exist_ok=True)


# Iterate through directory and count files
extensions = ('*.csv', '*.parquet')
file_count = 0
for ext in extensions:
	for file in parent_directory.rglob(ext):
		#print(file.name)
		file_count += 1
print("File Count: {}".format(file_count))

# Iterate through directory
extensions = ('*.csv', '*.parquet')
extensions = ('*.parquet', '*.csv')
#files_list = []
count = 1
for ext in extensions:
	for file in parent_directory.glob(ext):
		
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
		
		# Convert 'time' column to strings
		data['time'] = pd.to_numeric(data['time'].values)
		
		# Find separate flights
		data = clean_data(data)
		data, splits = find_splits(data, threshold)
		
		# Save split files
		if(len(splits) == 0):
			
			f = Path(str("{}_{}").format(file.stem, file.suffix))
			
			# Save file
			if(file.suffix == '.csv'):
				flights.get_group(group).to_csv(directory / f)
			elif(file.suffix == '.parquet'):
				flights.get_group(group).to_parquet(directory / f, allow_truncated_timestamps=True, engine = 'pyarrow')
			else:
				print("Invalid file extension.")
		else:
			data = label_flights(data, splits)
			flights = data.groupby('flight_number')
			
			subdir = directory / file.stem
			subdir.mkdir(parents=True, exist_ok=True)
			
			for group in flights.groups:
				
				f = Path(str("{}_{}{}").format(file.stem, int(group), file.suffix))
				
				# Save file
				if(file.suffix == '.csv'):
					flights.get_group(group).to_csv(subdir / f)
				elif(file.suffix == '.parquet'):
					flights.get_group(group).to_parquet(subdir / f, allow_truncated_timestamps=True)#, engine = 'pyarrow')
				else:
					print("Invalid file extension.")
					
		
exit()





data = pd.DataFrame()
splits = []
threshold = 900



data = pd.read_csv(directory / file)
data = clean_data(data)
data, splits = find_splits(data, threshold)
data = label_flights(data, splits)


print(data.columns)





exit()




directory = Path.cwd() / "data" / "test" / "flight_sep"
#file = "addf73.csv"
file = "a5681f.csv"


df = pd.read_csv(directory / file)

print(df.head())
print(df['icao24'].unique())

#df = df.drop_duplicates(subset = ["time", "lat", "lon", "geoaltitude"], keep = "last")

df = df.reset_index(drop = True)
print(df.index)
#time.sleep(10)

#df = df[:6517]
df['time'] = pd.to_datetime(df['time'],unit='s')

fig = px.scatter(df, x = df.index, y = "time", title = "[icao24: {}] Time by Index".format(df['icao24'].unique()[0]), hover_data={'index':(':.d', df.index), "time":':.d'})
fig.show()

# df = df.sort_values(by=['time'])
# df = df.reset_index(drop = True)
# fig = px.scatter(df, x = df.index, y = "time", title = "[icao24: {}] Time by Index".format(df['icao24'].unique()[0]), hover_data={'index':(':.d', df.index), "time":':.d'})
# fig.show()



fig = px.line(df, x = df.index, y = df['time'].diff(), title = "[icao24: {}] Time Diff by Index".format(df['icao24'].unique()[0]), hover_data={'index':(':.d', df.index), "time_diff":(':.d', df['time'].diff())})
fig.show()


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
	# elif(row['time_diff'] > mode):
	# 	splits.append(index)
	elif(row['time_diff'] > 900):
		splits.append(index)
		
print("Splits: {}".format(splits))

df.insert(df.shape[1], 'flight_number', None)
#print(df.columns)






exit()


# Label splits as separate flights/trips
prev_idx = 0
for i, val in enumerate(splits):
	
	if(i == 0):
		prev_idx = 0
	else:
		prev_idx = splits[i-1]
	
	print("[i = {}][prev_idx = {}][val = {}]".format(i, prev_idx, val))
	
	#print(df['time'][previous_index:val])
	#df['flight_number'][prev_idx:val] = i
	df.loc[prev_idx:val, 'flight_number'] = i
	
	#print(df['flight_number'][prev_idx:val])
	prev_idx = val
	
	if(i == len(splits)-1):
		#print("df.shape[1] = {}".format(df.shape[0]))
		#print("from i to end")
		#print("[i = {}]".format(i))
		prev_idx = val
		final_idx = df.shape[0]
		print("[i = {}][prev_idx = {}][val = {}]".format(i+1, prev_idx, val))
		
		df.loc[prev_idx:final_idx, 'flight_number'] = i+1
	
	
print("Unique: {}".format(df['flight_number'].unique()))













#print(df.groupby('flight_number').get_group(0))

df_grouped = df.groupby('flight_number')

for i in df_grouped.groups:
	_df = df_grouped.get_group(i)
	fig = px.scatter_3d(
		_df, 
		x = "lat", 
		y = "lon", 
		z = "geoaltitude", 
		color = _df['time'],
		range_color= [df['time'].min(), df['time'].max()],
		color_continuous_scale=px.colors.sequential.Inferno,
		title = "[icao24: {}][Flight/Trip #{}] Lat/Lon/Geoaltitude vs Time".format(_df['icao24'].unique()[0], i)
	)#, hover_data={'index':(':.d', df.index), "time":':.d'})
	# fig.update_xaxes(range=[df['lat'].min(), df['lat'].max()])
	# fig.update_yaxes(range=[df['lon'].min(), df['lon'].max()])
	# fig.update_zaxes(range=[df['geoaltitude'].min(), df['geoaltitude'].max()])
	fig.update_layout(
    scene = dict(
        xaxis = dict(range=[df['lat'].min(), df['lat'].max()],),
        yaxis = dict(range=[df['lon'].min(), df['lon'].max()],),
        zaxis = dict(range=[df['geoaltitude'].min(), df['geoaltitude'].max()],),)
	)
	
	fig.show()








import plotly.graph_objects as go


fig = go.Figure()
for i in df_grouped.groups:
	_df = df_grouped.get_group(i)
	fig.add_trace(go.Scattergeo(
        lon = _df['lon'],
        lat = _df['lat'],
		mode = 'markers',
		marker_color = df['geoaltitude'],
		name = i
	))
fig.update_geos(
	fitbounds="locations",
	showcountries = True,
	showsubunits = True,
	scope = 'usa', 
	resolution = 50,
	showlakes = True
	)
fig.update_layout(
    	title = "[icao24: {}] Lat/Lon vs Flight Number".format(df['icao24'].unique()[0]),
        #geo_scope='usa',
    )
fig.show()

fig.close_all
""" 
fig = go.Figure(data=go.Scattergeo(
        lon = df['lon'],
        lat = df['lat'],
		
        #text = df['text'],
        #mode = 'markers',
        #marker_color = df['flight_number'],
		mode = 'markers+lines',
		#legendgroup = "flight_number"
		
        ))
fig.update_geos(
	fitbounds="locations",
	showcountries = True,
	showsubunits = True,
	scope = 'usa', 
	resolution = 50,
	showlakes = True
	)
fig.update_layout(
    	title = "[icao24: {}] Lat/Lon vs Flight Number".format(df['icao24'].unique()[0]),
        #geo_scope='usa',
    )
fig.show() """










#outpath = dir / "nodupes.csv"
df.to_csv(Path(directory / Path("nodupes.csv")))