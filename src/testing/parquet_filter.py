import pandas as pd
from pathlib import Path

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



#Prepare arguments
input_directory = Path("D:\#FAA UAS Project\OpenSky WEEK\open_sky_data\data_parquets")
outfile = "states_2022-01-17-all.csv"

#Append files
#append_files(input_directory, outfile)

#Generate aircraft list
#aircraft = get_aircraft(input_directory)

#Filer aircraft list
#filter_aircraft(input_directory)

#Aggregate filtered aircraft
agg_filter_aircraft(input_directory, "states_2022-01-17-all_geofiltered.csv", "open_sky_dataaircraft_list_geofiltered_400to1000ft.txt")