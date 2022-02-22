import pandas as pd
from pathlib import Path

#Method to concatenate all files in a directory
def append_files(directory, output_filename):
	
	pathslist = directory.glob('**/*.csv')
	
	data = pd.DataFrame()
	
	for path in pathslist:
		df = pd.read_csv(path)
		data = data.append(df)
	
	print(df.shape)
	data.to_csv(Path( "output/" + output_filename), index = False)
	return


#Get all unique aircraft
def get_aircraft(directory):
	
	#Get list of file paths
	pathslist = directory.glob('**/*.csv')
	
	#Initialize set of aircraft
	aircraft = set([])
	
	#Initialize counter and loop through paths
	counter = 0
	file_count = len(list(directory.glob('**/*.csv')))
	for path in pathslist:
		print("Processing ({}/{}) '{}'".format(counter, file_count, path.name))
		counter = counter + 1
		
		#Load CSV
		df = pd.read_csv(path)
		
		#Add all unique craft to the crafts set
		for craft in df['icao24'].unique():
			aircraft.add(craft)
	
	#Save to a text file
	textfile = open(Path(Path.cwd() / "data/OpenSky/all_states 2022-01-17/aircraft_list.txt"), "w")
	for element in aircraft:
		textfile.write(element + "\n")
	textfile.close()
	
	#Return the set of aircraft
	print("Number of aircraft: " + str(len(aircraft)))
	return aircraft


#Modify the set to filter out groups with missing values
def filter_aircraft(directory):
	
	#Open aircraft list
	try:
		aircraft = set(line.strip() for line in open(Path(Path.cwd() / "data/OpenSky/all_states 2022-01-17/aircraft_list.txt")))
	except FileNotFoundError:
		print("ERROR: 'aircraft_list' not found. Please run the method to create it.")
		return
	
	""" print("Incoming...")
	print(list(aircraft))
	return """
	
	print("AC List Length (Before): " + str(len(list(aircraft))))
	
	
	#Get list of file paths
	pathslist = directory.glob('**/*.csv')
	
	#Initialize counter and loop through paths
	counter = 0
	file_count = len(list(directory.glob('**/*.csv')))
	for path in pathslist:
		print("Processing ({}/{}) '{}'".format(counter, file_count, path.name))
		counter = counter + 1
		
		df = pd.read_csv(path)
		groups = df.groupby('icao24')
		
		#Loop through groups
		for name, group in groups:
			
			if (name not in aircraft):
				continue
			
			###########
			# FILTERS #
			###########
			#Remove null
			if(group['geoaltitude'].isnull().values.sum() > 0):
				if (name in aircraft):
					aircraft.remove(name)
				
			#Altitude, 400 feet = 121.92 meters
			altitude = 122
			if(group['geoaltitude'].max() > altitude):
				if (name in aircraft):
					aircraft.remove(name)

	print("AC List Length (After): " + str(len(list(aircraft))))
	
	#Save to a text file
	textfile = open(Path(Path.cwd() / "data/OpenSky/all_states 2022-01-17/aircraft_list_geofiltered.txt"), "w")
	for element in aircraft:
		textfile.write(element + "\n")
	textfile.close()
	
	return aircraft
	
	

def agg_filter_aircraft(directory, outfile, aircraft):
	
	#Open aircraft list
	try:
		aircraft = set(line.strip() for line in open(Path(Path.cwd() / "data/OpenSky/all_states 2022-01-17" / aircraft)))
	except FileNotFoundError:
		print("ERROR: 'aircraft_list' not found. Please run the method to create it.")
		return
	
	print("Aircraft List Length: " + str(len(aircraft)))
	print(list(aircraft))
	
	#Initialize master dataframe
	data = pd.DataFrame()
	
	#Get list of file paths
	pathslist = directory.glob('**/*.csv')
	
	#Initialize counter and loop through paths
	counter = 1
	file_count = len(list(directory.glob('**/*.csv')))
	for path in pathslist:
		print("Processing ({}/{}) '{}'".format(counter, file_count, path.name))
		counter = counter + 1
		
		df = pd.read_csv(path)
		
		#data = data.append(df[df['icao24'].isin(aircraft)])
		df = df[df['icao24'].isin(aircraft)]
		if(df.empty):
			continue
		print(df)
		
		with open(Path("output/" + outfile), 'a') as f:
			df.to_csv(f, mode='a', header=f.tell()==0, line_terminator="\n")
		
		
	#print(data.shape)
	
	#data.to_csv(Path( "output/" + outfile), index = False)
	return



#Prepare arguments
input_directory = Path(Path.cwd() / "data/OpenSky/all_states 2022-01-17")
outfile = "states_2022-01-17-all.csv"

#Append files
#append_files(input_directory, outfile)

#Generate aircraft list
#aircraft = get_aircraft(input_directory)

#Filer aircraft list
#filter_aircraft(input_directory)

#Aggregate filtered aircraft
agg_filter_aircraft(input_directory, "states_2022-01-17-all_geofiltered.csv", "aircraft_list_geofiltered.txt")