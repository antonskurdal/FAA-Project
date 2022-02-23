import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
import seaborn as sns
import sys

pd.set_option('display.max_columns', None)


def calc_dropouts(infile, outfile, focus_col):
	"""Calculates the difference between the current column value and the previous for the specified column

	Args:
		infile (WindowsPath): Location of the input file
		outfile (WindowsPath): Location of the output file
		focus_col (string): column to analyze
	"""
	
	#Prepare data files
	df = pd.read_csv(infile)
	pdf = PdfPages(outfile)
	
	
	
	
	
	
	flightlist = list(df['flight'].unique())
	
	new_df = pd.DataFrame()
	
	######################################################################################################################################################
	for f in flightlist:
		
		#Get data for flight
		data = pd.DataFrame()
		data = df[df['flight'] == f].reset_index(drop = True).copy()
		
		#Dropout length
		data.insert(data.shape[1], 'dropout_length', data[focus_col].diff()[1:])
		
		#Mean
		data.insert(data.shape[1], 'flight_mean', data['dropout_length'][1:].mean())
		
		#Mode
		data.insert(data.shape[1], 'flight_mode', data['dropout_length'][1:].mode())
		data['flight_mode'] = data['flight_mode'][0]
		
		#Standard Deviation
		data.insert(data.shape[1], 'flight_stddev', data['dropout_length'][1:].std())

		#Z-Score
		zscores = stats.zscore(list(data['dropout_length'].dropna()))
		zscores = np.insert(zscores, 0, np.NaN, axis = 0)
		data.insert(data.shape[1], 'flight_zscores', zscores)
		
		print("Appending {} (Shape: {}".format(f, data.shape))
		#print("\tnew_df Shape Before: {}".format(new_df.shape))
		new_df = new_df.append(data)
		#print("\tnew_df Shape After: {}".format(new_df.shape))
	######################################################################################################################################################
	
	
	
	
	
	######################################################################################################################################################
	#Reset master index - NOTE: ALL NAN POINTS, WHICH ARE THE FIRST POINTS OF EACH CRAFT DATA SUBSET, ARE DROPPED HERE
	new_df = new_df.dropna()
	new_df.reset_index(inplace = True, drop = True)
	
	#Holistic Mean
	new_df.insert(new_df.shape[1], 'holistic_mean', new_df['dropout_length'].mean())
	
	#Holistic Mode
	new_df.insert(new_df.shape[1], 'holistic_mode', new_df['dropout_length'].mode())
	new_df['holistic_mode'] = new_df['holistic_mode'][0]
	
	#Holistic Standard Deviation
	new_df.insert(new_df.shape[1], 'holistic_stddev', new_df['dropout_length'].std())

	#Holistic Z-Score
	zscores = stats.zscore(list(new_df['dropout_length']))
	new_df.insert(new_df.shape[1], 'holistic_zscores', zscores)
	
	new_df.reset_index(inplace = True, drop = True)
	######################################################################################################################################################
	
	#Create metadata
	""" meta = new_df.describe()
	
	#Get mode of columns (when possible)
	modes = new_df.mode(axis = 0, numeric_only = True)
	for col in modes.columns:
		x = modes[col].notna().sum()
		if(x > 1):
			modes[col][0] = None
	meta = meta.append(modes.loc[[0]])
	meta = meta.rename(index = {0: 'mode'}) """
	
	
	#print(new_df['linear_acceleration_x'].mode())
	
	
	
	
	def describe(df, stats):
		d = df.describe()
		
		#Get mode of columns (when possible)
		modes = new_df.mode(axis = 0, numeric_only = True)
		for col in modes.columns:
			x = modes[col].notna().sum()
			if(x > 1):
				modes[col][0] = None
		d = d.append(modes.loc[[0]])
		d = d.rename(index = {0: 'mode'})
		
		return d.append(df.reindex(d.columns, axis = 1).agg(stats))
	
	#print(describe(new_df, ['skew', 'mad', 'kurt']))
	meta = pd.DataFrame()
	meta = describe(new_df, ['var', 'skew', 'kurt'])
	#print(meta)
	
	#print(new_df.describe())
	
	#Create distribution plot
	g = sns.displot(new_df, x = "flight_zscores", kind = "kde", hue = "flight", fill = True, legend = False)
	#ax.set(xlim=(-3.5,3.5))
	
	
	#Display figure
	plt.tight_layout()
	#plt.show()
	fig = g.fig
	pdf.savefig(fig)
	plt.close("all")
	
	
	
	
	
	
	
	
	
	
	
	#Close PDF
	pdf.close()
	
	#Save CSVs
	new_df.to_csv(Path(Path.cwd() / "output/DeliveryQuadcopter" / "flights_all_calc.csv"))
	meta.to_csv(Path(Path.cwd() / "output/DeliveryQuadcopter" / "flights_all_calc_meta.csv"))
	
	
	
	return
	
	
	
	
	bar_of_pie_zscore = False
	if (bar_of_pie_zscore == True):
			for group in categories.groups.keys():
				group_name = group
			
				#Set up group stuff
				#GROUPS
				# 'Heavy (> 300000 lbs)',
				# 'High Vortex Large (aircraft such as B-757)',
				# 'Large (75000 to 300000 lbs)',
				# 'Light (< 15500 lbs)',
				# 'No ADS-B Emitter Category Information',
				# 'Point Obstacle (includes tethered balloons)',
				# 'Small (15500 to 75000 lbs)'])
				#group_name = 'High Vortex Large (aircraft such as B-757)'
				wedge_num = list(categories_size_pct.keys()).index(group_name)
				
				group_data = categories.get_group(group_name)
				group_df = group_data[group_data['dropout_length'] > group_data['points_mode']]
				
				#WORK ON GETTING GROUP_DF TO REPLACE 'LIGHT' SO THIS CAN BE LOOPED
				
				#Create figure and axes
				fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6.5))
				fig.subplots_adjust(wspace = 0)
				
				#Pie chart parameters
				overall_ratios = list(categories_size_pct.values())
				labels = list(categories_size_pct.keys())
				
				overall_ratios.insert(0, overall_ratios.pop(wedge_num))
				labels.insert(0, labels.pop(wedge_num))
				
				
				#Create color palette
				from matplotlib.patches import ConnectionPatch
				import itertools
				pal = sns.color_palette(palette='tab10', n_colors=len(categories.groups))
				pal.insert(0, pal.pop(wedge_num))
				palette = itertools.cycle(pal)
				
				wedge_num = 0

				
				explode = [0, 0, 0, 0, 0, 0, 0]
				explode[wedge_num] = 0.1
				
				#Rotate so that first wedge is split by the x-axis
				if(overall_ratios[0] > 45):
					angle = 90 * overall_ratios[0]
				else:
					angle = 0#90 * overall_ratios[0]
				
				#Autopct
				def make_autopct(values):
					def my_autopct(pct):
						total = sum(values)
						val = int(round(pct*total/100.0))
						return '{p: .2f}%({v:d})'.format(p=pct,v=val)
					return my_autopct
				
				
				
				#Create pie
				wedges, labs, ax1_autotexts = ax1.pie(overall_ratios, autopct='%1.2f%%', startangle=angle, explode=explode, colors = pal, pctdistance = 0.8, wedgeprops = {'width' :0.4, 'linewidth' : 0, 'edgecolor': 'white'}, rotatelabels = True)
				plt.setp(ax1_autotexts, color = 'black', size=9, weight="bold")
				for label, pct_text in zip(labs, ax1_autotexts):
					pct_text.set_rotation(label.get_rotation())
				
				# Create a circle at the center of the plot
				my_circle = plt.Circle( (0,0), 0.7, color='white')
				
				#Pie Legend
				ax1.legend(labels = labels, bbox_to_anchor = (1, 0.1))
				
				#Add patch to show values in the center of the circle
				import matplotlib.patheffects as path_effects
				ax1.text(0, 0, "Mode:\n{:.2f}".format(int(group_data['dropout_length'].mode())), ha='center', va='center', fontsize=24, weight = 'normal', color = 'black')#.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
				
				#Bar chart parameters
				zscore_ratios = [
					group_df[(group_df['points_zscore'] > 0) & (group_df['points_zscore'] < 1)].reset_index(drop = True).copy().shape[0],		# > 0
					group_df[(group_df['points_zscore'] >= 1) & (group_df['points_zscore'] < 2)].reset_index(drop = True).copy().shape[0],		# >= 1
					group_df[(group_df['points_zscore'] >= 2) & (group_df['points_zscore'] < 3)].reset_index(drop = True).copy().shape[0],		# >= 2
					group_df[(group_df['points_zscore'] > 3)].reset_index(drop = True).copy().shape[0]											# > 3
				]
				zscore_ratios = list(map(lambda x: x / sum(zscore_ratios), zscore_ratios))
				#print(zscore_ratios)
				zscore_labels = [
					"0 to 0.9",
					"1 to 1.9",
					"2 to 2.9",
					"over 3"
				]
				bottom = 1
				width = .2
				
				#Adding from the top matches the legend
				for j, (height, label) in enumerate([*zip(zscore_ratios, zscore_labels)]):
					bottom -= height
					bc = ax2.bar(0, height, width, bottom=bottom, color='red', label=label,
								alpha=0.1 + 0.25 * j)
					ax2.bar_label(bc, labels=[f"{height:.3%}"], label_type='center', weight = 'bold')
				
				ax2.set_title('Dropout Severity')
				ax2.legend(title = "Z-Score", bbox_to_anchor = (0, -0.1375), loc = 'lower left')
				
				
				ax2.axis('off')
				ax2.set_xlim(- 2.5 * width, 2.5 * width)
				
				# use ConnectionPatch to draw lines between the two plots
				#print("WEDGE NUM: " + str(wedge_num))
				theta1, theta2 = wedges[wedge_num].theta1, wedges[wedge_num].theta2
				center, r = wedges[wedge_num].center, wedges[wedge_num].r
				bar_height = sum(zscore_ratios)
				
				# draw top connecting line
				x = r * np.cos(np.pi / 180 * theta2) + center[0]
				y = r * np.sin(np.pi / 180 * theta2) + center[1]
				con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
									xyB=(x, y), coordsB=ax1.transData)
				con.set_color([0, 0, 0])
				con.set_linewidth(1)
				ax2.add_artist(con)

				# draw bottom connecting line
				x = r * np.cos(np.pi / 180 * theta1) + center[0]
				y = r * np.sin(np.pi / 180 * theta1) + center[1]
				con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
									xyB=(x, y), coordsB=ax1.transData)
				con.set_color([0, 0, 0])
				ax2.add_artist(con)
				con.set_linewidth(1)
				
				#Set titles
				ax1.set_title("Total Number of Aircraft: {}\nNumber of Aircraft in Group: {}".format(len(points_master['icao24'].unique()), len(group_df['icao24'].unique())))
				plt.suptitle("Z-Score Breakdown (Points > Mode)\nAircraft Group: {}".format(group_name), weight = 'bold', fontsize = 12)
				
				#Display figure
				#plt.show()
				pdf.savefig(fig)
				plt.close("all")
		######################################################################################################################################################
		
	
	
	
	
	
	
	
	

directory = Path(Path.cwd() / "data/DeliveryQuadcopterDroneData/individual_flights")
infile = Path(Path.cwd() / "data/DeliveryQuadcopterDroneData" / "flights.csv")
outfile = Path(Path.cwd() / "output/DeliveryQuadcopter" / "flights_plots_all.pdf")



calc_dropouts(infile, outfile, "time")
