#!/usr/bin/env python

"""Clean and aggregate flight data
    
    description
"""

from pathlib import Path
import pandas as pd

__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2022, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"


# Set up directory & file
parent_directory = Path("D:/#FAA UAS Project/OpenSky WEEK/Individual Aircraft/batch/output")
file = parent_directory / "#AGG.csv"

# Delete file if it exists
if(file.is_file()):
	file.unlink()

# Open directory
extensions = ('*.parquet', '*.csv')
file_count = 0
for ext in extensions:
	for file in parent_directory.rglob(ext):
		file_count += 1
print("File Count: {}".format(file_count))

# Concatenate all files into a dataframe
parquet_files = [f for f in parent_directory.rglob('*.parquet')]
csv_files = [f for f in parent_directory.rglob('*.csv')]
parquet_df = pd.concat(map(pd.read_parquet, parquet_files), ignore_index = True)
csv_df = pd.concat(map(pd.read_csv, csv_files), ignore_index = True)
df = pd.concat([parquet_df, csv_df])

# Remove irrelevant columns
relevant_columns = ['time', 'taxonomy', 'icao24', 'lat', 'lon', 'geoaltitude', 'velocity', 'lastcontact', 'dropout_length']
df = df[relevant_columns]
print("Dataset Columns: {}".format(list(df.columns)))

# Drop invalid data
df = df.dropna(axis = 0, how = 'any', subset = ['lat', 'lon', 'geoaltitude', 'velocity', 'dropout_length', 'lastcontact'])

# Drop duplicates
df = df.drop_duplicates()
print("Number of Unique Aircraft: {}".format(len(df['icao24'].unique())))
print("Data Points Count: {}".format(df.shape[0]))

# Save aggregated file
file = file = parent_directory / "#AGG.csv"
print(file)
df.to_csv(file)