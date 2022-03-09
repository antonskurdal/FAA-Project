import multiprocessing
import pandas as pd
from pathlib import Path
import sys
import time
#Get all unique aircraft
def get_aircraft(directory):
	
	print("\nRETRIEVING UNIQUE AIRCRAFT")
	textfile = open(Path("D:\#FAA UAS Project\OpenSky WEEK\open_sky_data" + "aircraft_list.txt"), "w")
	#Get list of file paths
	pathslist = directory.glob('**/*.parquet')
	
	#Initialize set of aircraft
	aircraft = set([])
	
	#Initialize counter and loop through paths
	counter = 1
	file_count = len(list(directory.glob('**/*.parquet')))
	for path in pathslist:
		print("Processing ({}/{}) '{}'".format(counter, file_count, path.name))
		counter = counter + 1
		
		#Load CSV
		df = pd.read_parquet(path)
		
		#Add all unique craft to the crafts set
		for craft in df['icao24'].unique():
			aircraft.add(craft)
	
	#Save to a text file
	
	for element in aircraft:
		textfile.write(element + "\n")
	textfile.close()
	
	#Return the set of aircraft
	print("Number of aircraft: " + str(len(aircraft)))
	return aircraft


#Modify the set to filter out groups with missing values
def filter_aircraft(directory):
	
	print("\nFILTERING AIRCRAFT")
	
	#Open aircraft list
	try:
		aircraft = set(line.strip() for line in open(Path("D:\#FAA UAS Project\OpenSky WEEK\open_sky_data" + "aircraft_list.txt")))
	except FileNotFoundError:
		print("ERROR: 'aircraft_list' not found. Please run the method to create it.")
		return
	
	""" print("Incoming...")
	print(list(aircraft))
	return """
	
	print("AC List Length (Before): " + str(len(list(aircraft))))
	textfile = open(Path("D:\#FAA UAS Project\OpenSky WEEK\open_sky_data" + "aircraft_list_geofiltered_over1000ft.txt"), "w")
	
	#Get list of file paths
	pathslist = directory.glob('**/*.parquet')
	
	#Initialize counter and loop through paths
	counter = 1
	file_count = len(list(directory.glob('**/*.parquet')))
	for path in pathslist:
		print("Processing ({}/{}) '{}'".format(counter, file_count, path.name))
		counter = counter + 1
		
		df = pd.read_parquet(path)
		groups = df.groupby('icao24')
		
		#Loop through groups
		for name, group in groups:
			
			if (name not in aircraft):
				continue
			
			###########
			# FILTERS #
			###########
			#Remove null
			# if(group['geoaltitude'].isnull().values.sum() > 0):
			# 	if (name in aircraft):
			# 		aircraft.remove(name)
			
			"""
			#Altitude, [400 feet = 121.92 meters] > x
			if(group['geoaltitude'].max() > 122):
				if (name in aircraft):
					aircraft.remove(name)
			"""
			
			"""
			#Altitude, [400 feet = 121.92 meters] < x < [1000ft = 304.8 meters]
			if(group['geoaltitude'].max() < 122 or group['geoaltitude'].max() > 305):
				if (name in aircraft):
					aircraft.remove(name)
			"""
			
			#Altitude, x > [1000ft = 304.8 meters]
			if(group['geoaltitude'].max() < 305):
				if (name in aircraft):
					aircraft.remove(name)
	
	

	print("AC List Length (After): " + str(len(list(aircraft))))
	
	#Save to a text file
	
	for element in aircraft:
		textfile.write(element + "\n")
	textfile.close()
	
	return aircraft
	
	

def agg_filter_aircraft(directory, outfile, aircraft):
	
	#Open aircraft list
	try:
		aircraft = set(line.strip() for line in open(Path(Path("D:\#FAA UAS Project\OpenSky WEEK") / aircraft)))
	except FileNotFoundError:
		print("ERROR: 'aircraft_list' not found. Please run the method to create it.")
		return
	
	print("Aircraft List Length: " + str(len(aircraft)))
	print(list(aircraft))
	
	#Initialize master dataframe
	data = pd.DataFrame()
	
	#Get list of file paths
	pathslist = directory.glob('**/*.parquet')
	
	
	#Initialize counter and loop through paths
	counter = 1
	file_count = len(list(directory.glob('**/*.parquet')))
	for path in pathslist:
		print("Processing ({}/{}) '{}'".format(counter, file_count, path.name))
		counter = counter + 1
		
		df = pd.read_parquet(path)
		
		#data = data.append(df[df['icao24'].isin(aircraft)])
		df = df[df['icao24'].isin(aircraft)]
		if(df.empty):
			continue
		#print(df)
		
		with open(Path(Path("D:\#FAA UAS Project\OpenSky WEEK" + "agg_under400ft.csv")), 'a') as f:
			df.to_csv(f, mode='a', header=f.tell()==0, line_terminator="\n")
		
		
	#print(data.shape)
	
	#data.to_csv(Path( "output/" + outfile), index = False)
	return






###################################################################################################
#Gets flights from a text file and generates CSV files for each
def gen_flights(directory, output_directory, aircraft_list):
	import sys
	import time
	#Open aircraft list
	try:
		aircraft_list = set(line.strip() for line in open(Path(aircraft_list)))
		print("Aircraft List Length: " + str(len(aircraft_list)))
		#print(list(aircraft_list))
	except FileNotFoundError:
		print("ERROR: 'aircraft_list' not found. Please run the method to create it.")
		return
	
	
	
	_aircraft_list_len = len(list(aircraft_list))
	for i, craft in enumerate(list(aircraft_list)[:5]):
		#print(craft)
		
		#Get list of file paths
		pathslist = directory.glob('**/*.parquet')
		
		""" #Remove old files
		dir = Path("D:\#FAA UAS Project\OpenSky WEEK\Individual Aircraft")
		for f in dir.glob("**/*"):
			if(f.name in aircraft_list):
				print("Leaving '{}'".format(f.name))
				continue
			else:
				print("Deleting '{}'".format(f.name))
				f.unlink() """
		
		
		#Initialize dataframe
		craft_data = pd.DataFrame()
		
		#Loop through paths
		_file_count = len(list(directory.glob('**/*.parquet')))
		for j, path in enumerate(list(directory.glob('**/*.parquet'))):
			#print("Craft: [{}]({}/{}) - Parquat ({}/{}) '{}'".format(craft, i+1, _aircraft_list_len, j+1, _file_count, path.name))
			x = "Craft: [{}]({}/{}) - Parquat ({}/{}) '{}'".format(craft, i+1, _aircraft_list_len, j+1, _file_count, path.name)
			sys.stdout.write(x + "\n")
			#sys.stdout.flush()
			#time.sleep(0.2)
			
			
			df = pd.read_parquet(path)
			if(craft not in df['icao24'].unique()):
				continue
			
			
			craft_data = craft_data.append(df[df['icao24'] == craft])
			#craft_data.append()
			
			
			
			with open(Path(Path("D:\#FAA UAS Project\OpenSky WEEK\Individual Aircraft") / str(craft + ".csv")), 'a') as f:
				craft_data.to_csv(f, mode='a', header=f.tell()==0, line_terminator="\n")
			
	return
	
























def start_gen_flights_threaded(directory, output_directory, aircraft_list, num_threads):
	from multiprocessing import Pool
	
	if __name__ == '__main__':
		with Pool(processes=num_threads) as pool:
			nargs = [(directory, output_directory, aircraft_list, num_threads, n) for n in range(num_threads)]
			pool.starmap(gen_flights_threaded, nargs)
	
	
#def printstuff(directory, num_threads, threadnum):
		#print("Thread Number: {} - Craft Index: {}".format(threadnum, craftindex))
		

###################################################################################################
#THREADED - Gets flights from a text file and generates CSV files for each
def gen_flights_threaded(directory, output_directory, aircraft_list, num_threads, interval):
	
	#Open aircraft list
	try:
		aircraft_list = set(line.strip() for line in open(Path(aircraft_list)))
		print("Aircraft List Length: " + str(len(aircraft_list)))
		#print(list(aircraft_list))
	except FileNotFoundError:
		print("ERROR: 'aircraft_list' not found. Please run the method to create it.")
		return
	
	
	#aircraft_list = list(aircraft_list)
	aircraft_sublist = list(aircraft_list)[interval::num_threads]
	#print("\nThread Number: {}, Aircraft List Length: {}/{},\n\tList: {}".format(interval, before, after, aircraft_list))
	print("\nThread Number: {}, Aircraft List Length: ({}/{})".format(interval, len(aircraft_sublist), len(aircraft_list)))
	
	_aircraft_sublist_len = len(aircraft_sublist)
	for i, craft in enumerate(aircraft_sublist[:5]):
	
		#Check for existing files
		dir = Path("D:\#FAA UAS Project\OpenSky WEEK\Individual Aircraft")
		for f in dir.glob("**/*"):
			if(f.stem in aircraft_list):
				#print("Skipping '{}'".format(f.stem))
				continue
			# else:
			# 	print("Running '{}'".format(f.stem))
		
		#Get list of file paths
		pathslist = directory.glob('**/*.parquet')
		
		#Initialize dataframe
		craft_data = pd.DataFrame()
		
		#Loop through paths
		_file_count = len(list(directory.glob('**/*.parquet')))
		for j, path in enumerate(list(directory.glob('**/*.parquet'))):
			#print("Craft: [{}]({}/{}) - Parquat ({}/{}) '{}'".format(craft, i+1, _aircraft_list_len, j+1, _file_count, path.name))
			x = "[Thread {}] Craft: [{}]({}/{}) - Parquat ({}/{}) '{}'".format(interval, craft, i+1, _aircraft_sublist_len, j+1, _file_count, path.name)
			sys.stdout.write(x + "\n")
			sys.stdout.flush()
			time.sleep(0.2)
			
			
			df = pd.read_parquet(path)
			if(craft not in df['icao24'].unique()):
				continue
			
			
			craft_data = craft_data.append(df[df['icao24'] == craft])
			#craft_data.append()
			
			
			
			with open(Path(Path("D:\#FAA UAS Project\OpenSky WEEK\Individual Aircraft") / str(craft + ".csv")), 'a') as f:
				craft_data.to_csv(f, mode='a', header=f.tell()==0, line_terminator="\n")
			
			
	
	return
###################################################################################################


















#Prepare arguments
input_directory = Path("D:\#FAA UAS Project\OpenSky WEEK\open_sky_data\data_parquets")
outfile = "states_2022-01-17-all.csv"
threads = int(multiprocessing.cpu_count()/2)
print("Threads: ({}/{})".format(threads, multiprocessing.cpu_count()))
#Append files
#append_files(input_directory, outfile)

#Generate aircraft list
#aircraft = get_aircraft(input_directory)

#Filer aircraft list
#filter_aircraft(input_directory)

#Aggregate filtered aircraft
#agg_filter_aircraft(input_directory, "states_2022-01-17-all_geofiltered.csv", "open_sky_dataaircraft_list_geofiltered_400to1000ft.txt")

#Get flight CSVs
#gen_flights(input_directory, "D:\#FAA UAS Project\OpenSky WEEK\Individual Aircraft", "D:\#FAA UAS Project\OpenSky WEEK\opensky_week_aircraft_list.txt")

#THREADED - Get flight CSVs
start_gen_flights_threaded(input_directory, "D:\#FAA UAS Project\OpenSky WEEK\Individual Aircraft", "D:\#FAA UAS Project\OpenSky WEEK\opensky_week_aircraft_list.txt", threads)

#gen_flights_threaded(input_directory, "D:\#FAA UAS Project\OpenSky WEEK\Individual Aircraft", "D:\#FAA UAS Project\OpenSky WEEK\opensky_week_aircraft_list.txt", threads)