#!/usr/bin/env python

"""DFW Data Dropout Analyzer
    
    description
"""
import pandas as pd
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dateutil


__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2022, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"



#############
# LOAD DATA #
#############
# Set up directory
directory = Path.cwd() / "data" / "DFW"
file = directory / "ASSURE Data Assistance (UND).csv"
df = pd.read_csv(file)



##############
# CLEAN DATA #
##############
# Convert to epoch time - https://stackoverflow.com/questions/54313463/pandas-datetime-to-unix-timestamp-seconds
df['epoch_time'] = pd.to_datetime(df['Detection Time (EDT)']).map(pd.Timestamp.timestamp)

print(df['Flight ID'].unique())

# Group by flight ID
drones = df.groupby('Drone ID')
print(drones.groups)
print(drones.get_group(list(drones.groups)[0]))

#for drone in drones:
	#print(len(drone))
	
df['latency'] = df['epoch_time'].diff()

fig = px.scatter(df, x="epoch_time", y="latency")
fig.show()