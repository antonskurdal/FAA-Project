"""
############################
University of North Dakota
JSON Parser
10/28/2021

Developers: Anton Skurdal


Description:
Simple program to read JSON
files and print them in the
console.
############################

"""


import json
import csv
#import os
#import numpy as np
import time
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

pd.option_context('display.max_seq_items', None)
pd.option_context('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.max_colwidth',None)
#import matplotlib.pyplot as plt
#import seaborn as sns
#from mpl_toolkits.axes_grid1 import host_subplot
#from mpl_toolkits import axisartist

'''
#Make sure code runs as a module
if(__name__ == '__main__'):
	print("This code is meant to be run as a module.")
	exit(0)
'''

def get_individuals(path):
	
	floc = Path(path)
	print(floc)
	
	df = pd.read_csv(floc)#, nrows = 500000)
	#print(df.head(1))
	#print(df.columns)
	#data = df.copy()
	
	df = df.dropna()
	
	#Get list of unique tags and print them
	icao24_list = df['icao24'].unique()
	print("\nTAG LIST (FIRST 5):\n")
	print(*icao24_list[:5], sep = "\n")
	focus_tag = icao24_list[0]
	print("\nFOCUS TAG: " + focus_tag)
	
	print(len(icao24_list))
	icao24_list = icao24_list[:20]
	print(len(icao24_list))
	#return
	  
	#Get focus_tag data
	#all_tags = df.groupby('icao24')
	tag_data_grouped = df.groupby('icao24', as_index = False)
	
	
	pdf = PdfPages("icao24.pdf")
	i = 0
	for item in icao24_list:
		focus_tag = item
		print("Processing (" + str(i) + "/" + str(len(icao24_list)) + ")\n")
		i = i + 1
	
	
		tag_data = tag_data_grouped.get_group(focus_tag)
		tag_data = tag_data.reset_index(drop = True)
		
		#print(tag_data.head(5))
		df = tag_data
		
		#Remove Oct 31 2021 11:59:00 PM from timestamp
		#df['time'] = df['time'] - 1635724740
		df['time'] = df['time'] - 1635724740
		df['lastposupdate'] = df['lastposupdate'] - 1635724740
		df['lastcontact'] = df['lastcontact'] - 1635724740
		
		#df.astype({'time': 'int32'}, {'lastcontact': 'int32'}).dtypes
		#pd.to_numeric(df['time'], downcast = 'integer')
		#pd.to_numeric(df['lastcontact'], downcast = 'integer')
		df['lastcontact'] = df['lastcontact'].astype(int)
		df['lastposupdate'] = df['lastposupdate'].astype(int)
		
		data = df.copy()
		
		#drops = df['time'].diff()[1:]
		drops = df['time'] - df['lastcontact']
		
		drops2 = pd.DataFrame()
		drops2['time'] = df['time'] - df['lastposupdate']
		
		print(drops.head(5))
		#print(df.head(5))
		
		#Get rid of such drops
		#drops = drops.clip(lower = 0.9)
			
			
		
		#import seaborn as sns
		#import numpy as np
	
		
		
		
		data = data[data['icao24'].isin(icao24_list[:5])]
		
		
		data = data.groupby(['icao24'])
	
		
		
		
		df['time_lastcontact_diff'] = df['time'] - df['lastcontact']
		df['time_lastposupdate_diff'] = df['time'] - df['lastposupdate']
		
		data = df
		data = data[data['icao24'].isin(icao24_list[:5])]
		#data = data[data['icao24'] == icao24_list[0]]
	
		
	
	
	
		fig, axes = plt.subplots(2, 1)
		fig.set_size_inches(16, 8)
		
		#ax.plot(drops.index, drops)
		
		
		axes[0].plot(drops.index, drops)
		axes[1].plot(drops2.index, drops2)
		
		
		
		
		
	# ATTEMPTED COLOR MAP
	# =============================================================================
	# 	NbData = len(drops.index)
	# 	MaxBL = [[MaxBL] * NbData for MaxBL in range(100)]
	# 	Max = [np.asarray(MaxBL[x]) for x in range(100)]
	# 	
	# 	for x in range (0, 2):
	# 	  axes[0].fill_between(drops.index, Max[x], drops, where=drops >=Max[x], facecolor='green', alpha=0.3)
	# 	
	# 	for x in range (0, 3):
	# 	  axes[0].fill_between(drops.index, drops, Max[x], where=drops <Max[x], facecolor='red', alpha=0.3)
	# 	
	# 	axes[0].fill_between([], [], [], facecolor='red', label="x > 50")
	# 	plt.fill_between([], [], [], facecolor='green', label="x < 50")
	# =============================================================================
		
		
		
		
		
		
		
		
		
		
		
		# Set common labels
		fig.suptitle("icao24:" + focus_tag + " - Seconds Since Contact & Position Update \nAbove 1.0 = longer than normal response time (likely a dropout)")
		axes[0].set_xlabel('Time Index (Data Point)')
		axes[1].set_xlabel('Time Index (Data Point)')
		
		axes[0].set_ylabel('Time Since Last Contact (seconds)')
		axes[1].set_ylabel('Time Since Last Position Update (seconds)')
		
		
		
		
		
		#plt.xlabel("Data Index")
		#plt.ylabel("Time Since Last Position Update (seconds)")
		#plt.title("icao24: " + focus_tag + " Dropout Length by Index\n")
	
	
	
	
	
	
	
	
		
		
		
		
		
		
		
		
		#plt.tight_layout()
		#plt.show()
		#plt.close("all")
		pdf.savefig(fig)
		plt.close("all")
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		#CREATE PDF FOR EACH FOCUS TAG IN LOOP
		
		#fig = plt.figure(figsize=(10,5))
		#ax = plt.axes()
		
		fig, axes = plt.subplots(2, 1)
		fig.set_size_inches(16, 8)
		fig.suptitle("icao24:" + focus_tag + " - Seconds Since Contact & Position Update  - Density Distributon")
		
		#sns.kdeplot(df.loc[df['icao24'] == focus_tag, "time_lastposupdate_diff"], shade=True, color="b", label="Cyl=4", alpha=.7, ax=axes[0])
		sns.kdeplot(df.loc[df['icao24'] == focus_tag, "time_lastposupdate_diff"], shade=True, color="b", alpha=.5, ax=axes[0])
		
		sns.kdeplot(df.loc[df['icao24'] == focus_tag, "time_lastcontact_diff"], shade=True, color="r", alpha=.5, ax=axes[1])
		
		
		
		
		
		
		
		#axes[0].set(xscale="log", yscale="log")
		
		#Function x**(1/2)
		def forward(x):
			return x**(1/2)
		
		def inverse(x):
			return x**2
		
		
		axes[0].set_yscale('function', functions=(forward, inverse))
		axes[0].yaxis.set_major_locator(plt.FixedLocator(np.arange(0, 13, 1)**2))
		axes[0].yaxis.set_major_locator(plt.FixedLocator(np.arange(0, 13, 1)))
		
		axes[1].set_yscale('function', functions=(forward, inverse))
		axes[1].yaxis.set_major_locator(plt.FixedLocator(np.arange(0, 13, 1)**2))
		axes[1].yaxis.set_major_locator(plt.FixedLocator(np.arange(0, 13, 1)))
		
		#axes[0].set_xlim(-1,3)
		#axes[1].set_xlim(-1,3)
		
		axes[0].set_xlabel('Time Since Last Contact (seconds)')
		axes[1].set_xlabel('Time Since Last Position Update (seconds)')
		
		
		
		#plt.ylabel("Time Since Last Position Update (seconds)")
		#plt.title("icao24: " + focus_tag + " Seconds Since Contact & Position Update Distributon\n")
		
		#plt.tight_layout()
		#plt.show()
		
		pdf.savefig(fig)
		plt.close("all")
		
	pdf.close()
	return
	
	
	


def opensky_dropout_sandbox(path):
	
	
	#floc = Path(Path().resolve().parents[0] / path)
	floc = path
	print(floc)
	
	
	
	
	
	#sensorNum = str(Path(path).name)[8:-4]
	#print(sensorNum)
	
	
	df = pd.read_csv(path)#, nrows = 500000)
	#print(df.head(1))
	print(df.columns)
	
	
	#return
	
	
	
	
	drops = df['timestamp'].diff()[1:]
	print(drops.head(5))
	print(df['timestamp'].head(5))
	
	import matplotlib.pyplot as plt
	
	fig = plt.figure()
	ax = plt.axes()
	
	#x = np.linspace(0, 10, 1000)
	ax.plot(drops.index, drops)
	
	plt.xlabel("Time Index (Message Number)")
	plt.ylabel("Time Between Messages (Delay Length)")
	plt.title("OpenSky Sensor SN#" + sensorNum + " Dropout Length by Index\n")
	
	plt.show()

def opensky_sandbox(path):
	
	sensorListIndex = 5;
	
	#floc = Path(Path().resolve().parents[0] / path)
	floc = path
	print(floc)
	
	df = pd.read_csv(path)#, nrows = 500000)
	#print(df.head(1))
	serial_nums = df['sensorSerialNumber'].unique()
	print(serial_nums)
	
	serial = df.groupby('sensorSerialNumber')
	serial = serial.get_group(serial_nums[sensorListIndex]).reset_index(0, drop = True)
	print(serial['sensorSerialNumber'][sensorListIndex])
	
	#return
	path = Path(path)
	#print(path.parent)
	csv_name = Path(path.parent / Path("opensky_" + str(serial['sensorSerialNumber'][sensorListIndex]) + ".csv"))
	print(csv_name)
	serial.to_csv(csv_name, index = False)	
	
	
	return

def avro_csv_sandbox(path):
	
	#floc = Path(Path().resolve().parents[0] / path)
	floc = path
	print(floc)
	
	df = pd.read_csv(floc, nrows = 500000)
	#print(df.head(1))
	print(df['sensorType'].unique())
	
	csv_name = Path(str(floc)[:-4] + "_sandbox.csv")
	#df.to_csv(csv_name)
	
	opensky = df.groupby('sensorType')
	opensky = opensky.get_group('OpenSky').reset_index(0, drop = True)
	print(opensky)
	
	path = Path(path)
	print(path.parent)
	csv_name = Path(path.parent / Path("opensky.csv"))
	#opensky.to_csv(csv_name, index = False)
	
	return

def avro_csv_trim(path):
	
	#floc = Path(Path().resolve().parents[0] / path)
	floc = path
	print(floc)
	
	df = pd.read_csv(floc, nrows = 500000)
	#print(df.head(1))
	#print(df['sensorType'].unique())
	csv_name = Path(str(floc)[:-5] + "_500k.csv")
	df.to_csv(csv_name)
	#print(csv_name)	
	#time.sleep(5)
	
	return

def avro_test2(path):
	from fastavro import writer, reader, parse_schema
	
	floc = Path(Path().resolve().parents[0] / path)
	print(floc)
	
	"""
	# Reading
	with open(floc, 'rb') as fo:
		for record in reader(fo):
			print(record)
	"""
	
	csv_name = Path(str(floc)[:-5] + "_M.csv")
	print(csv_name)
	
	
	
	head = True
	count = 0
	f = csv.writer(open(csv_name, "w+"))
	with open(floc, 'rb') as fo:
		avro_reader = reader(fo)
		for emp in avro_reader:
			#print(emp)
			if head == True:
				header = emp.keys()
				f.writerow(header)
				head = False
			count += 1
			f.writerow(emp.values())
	print(count)	
	
	
	return

def avro_test(path):
	import avro.schema as avsc
	import avro.datafile as avdf
	import avro.io as avio
	
	floc = Path(Path().resolve().parents[0] / path)
	print(floc)
	
	#reader_schema = avsc.parse(open("reader.avsc", "rb").read())
	reader_schema = avsc.parse(open(floc, "rb").read())
	
	# need ability to inject reader schema as 3rd arg
	#with avdf.DataFileReader(open("record.avro", "rb"), avio.DatumReader()) as reader:
	with avdf.DataFileReader(open(floc, "rb"), avio.DatumReader()) as reader:
		for record in reader:
			print(record)
	
	return

def json_csv(path):
	
	print(Path().resolve().parents[0])
	
	#floc = Path(Path.cwd() / path)
	floc = Path(Path().resolve().parents[0] / path)
	
	print(floc)
	
	#directory = str(os.path.join(os.getcwd(), folder))
	
	with open(floc, "r") as read_file:
		data = json.load(read_file)
	print(data.keys())
	print(data['meta'])
	
	
	csv_name = Path(str(floc)[:-5] + "_M.csv")
	print(csv_name)
	
	#df = pd.read_json(floc)
	#print(df)
	#df.to_csv (directory +.csv', index = None)
	

file = "data\\test\\test.json"
#json_csv(file)

avro_file = "data\\opensky-network\\20170109_16_anonymized.avro" #Change this to drive (D:)
#avro_test2(avro_file)

avro_ref = "D:\\FAA UAS Project\\20170109_16_anonymized_M.csv"
#avro_csv_trim(avro_ref)

avro_trim = "D:\\FAA UAS Project\\20170109_16_anonymized__500k_sandbox.csv"
#avro_csv_sandbox(avro_trim)

opensky_name = "D:\\FAA UAS Project\\opensky_sandbox.csv"
#opensky_sandbox(opensky_name)

#opensky_dropout_name = "D:\\FAA UAS Project\\opensky_954778341_dropout.csv"
opensky_dropout_name = "D:\\FAA UAS Project\\AircraftDatabase\\aircraftDatabase-2020-11.csv"
#opensky_dropout_sandbox(opensky_dropout_name)

floc = "D:\\FAA UAS Project\\AircraftDatabase\\states_2021-11-01-00.csv"
get_individuals(floc)


print("\nDone")