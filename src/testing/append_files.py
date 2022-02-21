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


#Prepare arguments
input_directory = Path(Path.cwd() / "data/OpenSky/all_states 2022-01-17")
outfile = "states_2022-01-17-all.csv"

#Run method
append_files(input_directory, outfile)

