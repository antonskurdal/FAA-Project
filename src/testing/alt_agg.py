import pandas as pd
from pathlib import Path
import numpy as np






def icao_matching(data):



	#Format old columns
	data['icao24'].astype(str)
	data['icao24'] = data['icao24'].str.lower()
	
	#Create new columns
	data['found'] = False
	data['found'].astype(bool)
	data['manufacturername'] = ""
	data['manufacturername'].astype(str)
	data['model'] = ""
	data['model'].astype(str)
	
	data['typecode'] = ""
	data['typecode'].astype(str)
	data['icaoaircrafttype'] = ""
	data['icaoaircrafttype'].astype(str)
	data['categoryDescription'] = ""
	data['categoryDescription'].astype(str)
	#print(data.head(5))
	
	#Open database
	focus_cols = ['icao24', 'manufacturername', 'model',  'typecode', 'icaoaircrafttype', 'categoryDescription']
	database = pd.read_csv(Path(Path.cwd() / "data/databases/OpenSky/aircraftDatabase-2022-02.csv"), low_memory = False, usecols = focus_cols)
	#print(database.head(5))
	#print(database.dtypes)
	
	database['icao24'].astype(str)
	database['icao24'] = database['icao24'].str.lower()
	
	
	craftlist = data['icao24'].unique()
	#modeslist = database['icao24'].unique()
	#print(modeslist)
	database['icao24'] = database['icao24'].str.rstrip()
	#data['icao24'] = data['icao24'].str.upper()
	
	#agg = pd.DataFrame()
	
	#print(database['icao24'])
	
	
	
	loopsize = 250
	data = data[data['icao24'].isin(craftlist[:loopsize])]
	
	for count, craft in enumerate(craftlist[:loopsize]):
	#for count, craft in enumerate(craftlist[:10]):
		
		#print(craft)
		#print(craft.upper())
		#craft = craft.upper()
		
		print("\n(" + str(count) + "/" + str(len(craftlist)) + ")" + " Checking ICAO24: " + str(craft))
		
		start_time = time.time()
		
		#db = database[database['icao24'] == craft].copy()
		db = database.loc[(database['icao24'] == craft)]
		
		#print(db)
		
		#Check
		if(db.empty == False):
			
			#Print(db)
			
			#Works and is fast
			data.loc[(data['icao24'] == craft), ['found', 'manufacturername', 'model', 'typecode', 'icaoaircrafttype', 'categoryDescription']] = [
				True,
				db['manufacturername'].values[0],
				db['model'].values[0],
				db['typecode'].values[0],
				db['icaoaircrafttype'].values[0],
				db['categoryDescription'].values[0],
			]
			
			#Works, but slower
			""" data.loc[(data['icao24'] == craft), 'found'] = True
			data.loc[(data['icao24'] == craft), 'manufacturername'] = db['manufacturername'].values[0]
			data.loc[(data['icao24'] == craft), 'model'] = db['model'].values[0]
			data.loc[(data['icao24'] == craft), 'typecode'] = db['typecode'].values[0]
			data.loc[(data['icao24'] == craft), 'icaoaircrafttype'] = db['icaoaircrafttype'].values[0]
			data.loc[(data['icao24'] == craft), 'categoryDescription'] = db['categoryDescription'].values[0] """
			
			
			
			print("[TRUE] Finished in {:.4f} seconds".format(time.time() - start_time))
		else:
			
			#data = data.loc[(data['found'] == True)]
			
			
			#Drop rows with column 'icao24' equal to craft (both work)
			#print("pre: " + str(data.shape))
			#data = data.drop(data[data.icao24 == craft].index).copy()
			data = data.loc[(data['icao24'] != craft)]
			data.reset_index(drop = True, inplace = True)
			#print("post: " + str(data.shape))
			
			print("[FALSE] Finished in {:.4f} seconds".format(time.time() - start_time))
		
		
	print("\n[FOUND] Uniques: " + str(data['found'].unique()))
	
	#data = data.loc[(data['found'] == True)]
	data = data.drop(columns = ['found'], axis = 1)
	data = data[data['categoryDescription'].notna()]
	data.replace('', np.nan, inplace = True)
	data.dropna()
	data.reset_index(drop = True, inplace = True)
	
	
	print("\n[CATEGORY DESCRIPTION] Uniques: " + str(data['categoryDescription'].unique()))
	
	
	data.to_csv("data/OpenSky/output/states_2022-01-17-all_agg.csv", index = False)
	
	

def altitude_extraction(infile, outfile, altitude):
	"""Grabs all rows from a CSV where geoaltitude is less than the specified height.
	
	baroaltitude/geoaltitude
		These two columns indicate the aircraft's altitudel. As the names suggest, baroaltitude
		is the altitude measured by the barometer and depends on factors such as weather, whereas
		geoaltitude is determined using the GNSS (GPS) sensor. In our case, the aircraft was
		flying at a geometric altitude (or height) of 9342.12 meters and a barometric altitude
		of 9144 meters. That makes a difference of almost 200 meters. You are likely to observe
		similar differences for aircraft in spatial and temporal vicinity. Note that due to its
		importance in aviation, barometric altitude will almost always be present, while the
		geometric altitude depends on the equipage of the aircraft.
	
	
	
	Args:
		infile (_type_): _description_
		outfile (_type_): _description_
		altitude (_type_): _description_
	"""
	
	#Read in file
	df = pd.read_csv(infile)
	
	#Group by icao24
	groups = df.groupby('icao24')
	
	
	print("Before groups:" + str(groups.ngroups))
	groups = groups.filter(lambda g: g.isnull().values.any() == False)
	print("After groups:" + str(groups.groupby('icao24').ngroups))
	
	print(type(groups))
	
	
	groups.to_csv(Path(Path.cwd() / "output/alt_agg" / "states_2022-01-17-all_notna.csv"))
	
	
	return
	
	
	
	df.loc[df.groupby('id')['val'].filter(lambda x: len(x[pd.isnull(x)] ) < 2).index]

	
	
	
	
	
	
	
	
	
	
	print("\nColumns:")
	print(df.columns)
	
	print("\nData Types:")
	print(df.dtypes)
	
	print("\nMemory Usage (bytes):")
	print(df.memory_usage(deep=True))
	
	
	df2 = df.copy()
	
	df2['icao24'] = df2['icao24'].astype("category")
	df2['callsign'] = df2['callsign'].astype("category")
	
	df2[['time', 'lat', 'lon', 'velocity', 'heading', 'vertrate', 'squawk', 'baroaltitude', 'geoaltitude', 'lastposupdate', 'lastcontact']] = df2[['time', 'lat', 'lon', 'velocity', 'heading', 'vertrate', 'squawk', 'baroaltitude', 'geoaltitude', 'lastposupdate', 'lastcontact']].apply(pd.to_numeric, downcast="float")
	print("\nMemory Usage 2 (bytes):")
	print(df2.memory_usage(deep=True))
	
	
	reduction = 1 - (df2.memory_usage(deep=True).sum() / df.memory_usage(deep=True).sum())
	print("\nReduction: {:.1f}%".format(reduction*100))
	
	#return
	
	
	
	
	
	
	
	out_df = pd.DataFrame()
	outfile.unlink(missing_ok = True)
	
	
	craftslist = df['icao24'].unique()
	
	groups = df.groupby('icao24')
	
	#Working filter with small datasets (<1million rows)
	""" dfnew = df.groupby('icao24').filter(lambda x: x['geoaltitude'].max() < altitude and not(x.isnull().values.any()))
	dfnew.reset_index(drop=True, inplace=True) # reset index
	print(dfnew)
	dfnew.to_csv(outfile)
	
	return """
	
	
	
	
	for i, craft in enumerate(craftslist):
		print("Progress: ({}/{})".format(i, len(craftslist)))
		x = groups.get_group(craft)
		
		if((max(x['geoaltitude'] > altitude)) or (x.isnull().values.any())):
			continue
		
		
		x.reset_index(level=0, inplace=True, drop = True)
		
		with open(outfile, 'a') as f:
			x.to_csv(f, mode='a', header=f.tell()==0)
		
		
		#x.to_csv(outfile, mode = 'a', header = True)
		
		
		
		#out_df = out_df.append(x)
		
		# if(outfile.is_file() == False):
		# 	x.to_csv(outfile, mode = 'a', header = True)
		# else:
		# 	x.to_csv(outfile, mode = 'a', header = False)
	
	
	
	
	
	
	
	
	
	
	
	
	""" out = pd.read_csv(outfile)
	print(out.columns)
	out = out.rename(columns = {'Unnamed: 0': "message_index"})
	out.index.name = 'index'
	out.to_csv(outfile) """
		
	#out_df.reset_index(inplace = True)
	#out_df.to_csv(outfile)
	
	
	return
	
	
	
	
	
	df = df[df['geoaltitude'] < altitude]
	
	
	df.to_csv(outfile)
	
	
	

def remove_na_groups(infile, outfile):
	
	#Read in file
	df = pd.read_csv(infile)
	
	#Group by icao24
	groups = df.groupby('icao24')
	
	#Filter out na values
	print("Before groups:" + str(groups.ngroups))
	groups = groups.filter(lambda g: g.isnull().values.any() == False)
	print("After groups:" + str(groups.groupby('icao24').ngroups))
	print(type(groups))
	
	#Save to file
	groups.to_csv(outfile)
	return



remove_na_groups(Path(Path.cwd() / "data/OpenSky/" / "states_2022-01-17-10.csv"), Path(Path.cwd() / "output/alt_agg" / "states_2022-01-17-all_notna.csv"))
exit(0)

#Set up input/output files
infile = Path(Path.cwd() / "data/OpenSky/" / "states_2022-01-17-10.csv")
outfile = Path(Path.cwd() / "output/alt_agg" / "states_2022-01-17-10_LT400ft.csv")

#Altitude, 400 meters = 121.92 feet
altitude = 122

#Call method
altitude_extraction(infile, outfile, altitude)