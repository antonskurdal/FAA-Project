from matplotlib import docstring
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from pyparsing import alphas
import scipy.stats as stats
import numpy as np
import seaborn as sns
import time
# sns.set_style('darkgrid', {"axes.facecolor": "lightgray"})
# sns.dark_palette("seagreen", as_cmap=True)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None  # default='warn'


'''
#Shade areas - WORKING
# Get the lines and xy data from the lines so that we can shade
x1 = ax.ax.get_lines()[0].get_xydata()[:,0]
y1 = ax.ax.get_lines()[0].get_xydata()[:,1]
#ax.ax.fill_between(x1,y1, color="red", alpha=0.3)
plt.axvspan(-1, 0, color='#009A44', alpha=0.6, lw=0)
plt.axvspan(0, 1, color='r', alpha=0.6, lw=0)
ax.ax.fill_between(x1,y1, np.max(y1), color="white", alpha=1)
'''
'''
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

x = x1
dx = x[1]-x[0]
y = y1
dydx = np.gradient(y, dx)  # first derivative

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
'''
'''
#Can't figure it out
# Create a continuous norm to map from data points to colors
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(2)
line = ax.ax.add_collection(lc)
#fig.colorbar(line, ax=ax.ax)
# Use a boundary norm instead
cmap = ListedColormap(['r', 'g', 'b'])
norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N, clip=True)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(dydx)
lc.set_linewidth(2)
line = ax.ax.add_collection(lc)
'''
""" #WORKING Y-Gradient
xData = x1
yData = y1
NbData = len(xData)
MaxBL = [[MaxBL] * NbData for MaxBL in range(100)]
Max = [np.asarray(MaxBL[x]) for x in range(100)]

for x in range (50, 100):
	ax.ax.fill_between(xData, Max[x], yData, where=yData >Max[x], facecolor='red', alpha=0.02)

for x in range (0, 50):
	ax.ax.fill_between(xData, yData, Max[x], where=yData <Max[x], facecolor='green', alpha=0.02) """


""" #WORKING HISTOGRAM BIN LABELS - NOT TO SCALE
			s = 0
			for p in ax.ax.patches:
				s+= p.get_height()

			for p in ax.ax.patches: 
				ax.ax.text(p.get_x() + p.get_width()/2.,
						p.get_height(),
						'{}'.format(int(p.get_height()*100/s)), 
						fontsize=14,
						color='red',
						ha='center',
						va='bottom')
			 """
#Set working path
import sys
base_dir = Path.resolve(Path.cwd())
sys.path.insert(0, base_dir)
""" print("\nBASE DIR: " + str(base_dir)) """


#Calculate Dropouts - organizes data, creates a pdf of plots, saves csv
def calc_dropouts(data, outfile, quantity):
	if(isinstance(quantity, list) == False):
		quantity = str(quantity)
		quantity = map(int, quantity)
		quantity = list(quantity)
	
	print(quantity)

	metadata = pd.DataFrame(columns = ['num_craft', 'num_points', 'avg', 'mode', 'std_dev', 'min', 'max'])

	for q in quantity:
		print("Q: " + str(q))

		#Format lastcontact to not be in scientific notation
		data['lastcontact'] = data['lastcontact'].astype(int)

		#Create a master dataframe
		points_master = pd.DataFrame()

		#Get unique aircraft and truncate to input number
		craft_list = data['icao24'].unique()
		
		if(q <= len(craft_list)):
			craft_list = craft_list[:q]
		
		print("Length of craft list: " + str(len(craft_list)))
		#print("LENGTH OF CRAFT LIST [Q]: " + str(len(craft_list)))

		#Iterate over craft list
		for craft in craft_list:

			#Get data for craft
			points = pd.DataFrame()
			points = data[data['icao24'] == craft].reset_index(drop = True).copy()

			#Dropout length
			points.insert(points.shape[1], 'dropout_length', points['lastcontact'].diff()[1:])

			#Mean
			points.insert(points.shape[1], 'points_avg', points['dropout_length'][1:].mean())

			#Mode
			points.insert(points.shape[1], 'points_mode', points['dropout_length'][1:].mode())
			points['points_mode'] = points['points_mode'][0]
			
			#Standard Deviation
			points.insert(points.shape[1], 'points_stddev', points['dropout_length'][1:].std())

			#Z-Score
			zscores = stats.zscore(list(points['dropout_length'].dropna()))
			zscores = np.insert(zscores, 0, np.NaN, axis = 0)
			points.insert(points.shape[1], 'points_zscore', zscores)
			#print(points['dropout_zscore'])

			#Rounded Z-Score
			zscores_round = points.points_zscore.mul(2).round().div(2)
			points.insert(points.shape[1], 'points_zscore_round', zscores_round)
			#print(zscores_round)
			#print(points['dropout_zscore_round'])
			#print(points)

			points_master = points_master.append(points)
			#print(points_master)

			draw_plots = False

			if(draw_plots == True):


				'''
				# Draw Plot
				
				import joypy
				fig, axes = joypy.joyplot(points, by="icao24", column="dropout_zscore", ylim='own')
				for a in axes[:-1]:
					a.margins(3)
					a.set_xlim([-4,4])
				
				
				#fig, ax = plt.subplots()
				#ax.hist(points['dropout_zscore'], range = [-3,3])	#Working histogram
				#ax = points['dropout_zscore'].plot.kde()				#Working kdeplot
				'''

				#Create distribution plot
				ax = sns.displot(points, x = "points_zscore", kind = "kde", hue = "icao24", fill = True)
				ax.set(xlim=(-3.5,3.5))
				plt.show()

				'''
				#Calculate dropout percentages
				num_points = points.shape[0]
				num_dropouts_avg = num_points - points[(points['dropout_length'] > points['points_avg'][0])].shape[0]
				pct_dropouts_avg = num_points/num_dropouts_avg
				'''

				#Number of points
				num_points = points.shape[0]

				#Number of points greater than their average
				num_dropouts_avg = points[(points['dropout_length'] > points['points_avg'])].reset_index(drop = True).copy().shape[0]
				pct_dropouts_avg = num_dropouts_avg/num_points * 100

				#Number of points greater than their mode
				num_dropouts_mode = points[(points['dropout_length'] > points['points_mode'])].reset_index(drop = True).copy().shape[0]
				pct_dropouts_mode = num_dropouts_mode/num_points * 100

				#Create plot
				fig, ax = plt.subplots()
				ax.plot(points.index, points['dropout_length'], linestyle = '-', color = 'gray', zorder = 1)
				ax.scatter(points.index, points['dropout_length'], c=cm.RdYlGn_r(points['dropout_length']/points['dropout_length'].max()), zorder = 2, edgecolors = 'gray')
				ax.set_ylim(ax.get_ylim()[::-1])

				#Add color bar
				norm = mpl.colors.Normalize(vmin=0, vmax=1000000)
				cmap = plt.cm.RdYlGn
				cax = fig.add_axes([0.92, 0.12, 0.02, 0.75])
				cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, spacing='proportional')
				cb.set_label('Dropout Severity')
				cb.set_ticks([])

				#Add average line
				ax.axhline(points['points_avg'][0], xmin = 0, xmax = 1, label='Average: {:0.3f}'.format(points['points_avg'][0]), linestyle='--', color = 'black')
				#fig.canvas.draw_idle()  # use draw_idle instead of draw
				
				#Add mode line
				ax.axhline(points['points_mode'][0], xmin = 0, xmax = 1, label='Mode: {:0.3f}'.format(points['points_mode'][0]), linestyle='--', color = 'blue')

				
				
				#Add other information
				ax.legend()
				ax.set_xlabel("Packet Number (Index)")
				ax.set_ylabel("Time Since Last Contact (seconds)\n[Reversed Axis]")
				plt.suptitle(" icao24: " + str(craft) + " Dropouts [Points > Average]")
				title = "Points: " + str(num_points)
				#title = ("Points: " + str(num_points) + "\tDropouts: " + str(num_dropouts_avg) + "\tPercent Dropouts: {:0.4f}%".format(pct_dropouts_avg)).expandtabs()
				ax.set_title(title)

				plt.show()
				return

				#Save figure
				#pdf.savefig(fig)
				#plt.close("all")
		




		######################################################################################################################################################
		#Reset master index - NOTE: ALL NAN POINTS, WHICH ARE THE FIRST POINTS OF EACH CRAFT DATA SUBSET, ARE DROPPED HERE
		#print(points_master.shape)
		
		#x = points_master.groupby('icao24')
		num_craft = points_master['icao24'].unique().shape[0]



		#print("\nX SIZE PRE-DROP: " + str(num_craft) + "\n" + str(x.size()))
		print("Number of Aircraft [PRE]: " + str(num_craft))
		# num_craft = points_master['icao24'].unique().shape[0]
		# print("Number of Aircraft: " + str(num_craft))
		#points_master = points_master.dropna()

		#x = points_master.groupby('icao24')
		#print("\nX SIZE POST-DROP: " + str(num_craft) + "\n" + str(x.size()))


		#print(points_master.shape)
		
		
		#print(points_master.head(5))
		points_master.reset_index(inplace = True)
		print(points_master.head(5))


		'''
		#Reset master dataframe index
		points_master.reset_index(inplace=True)
		'''
		#print(points_master)
		#print(points_master['points_mode'].unique())

		#Number of unique aircraft
		
		num_craft = points_master['icao24'].unique().shape[0]
		print("Number of Aircraft [POST]: " + str(num_craft))

		#Calculate average of all averages
		#Calculate mode of all modes

		#Number of points
		num_points = points_master.shape[0]

		#Number of points greater than average - NOT MASTER AVERAGE, EACH AIRCRAFT HAS THEIR OWN
		num_dropouts_avg = points_master[(points_master['dropout_length'] > points_master['points_avg'])].reset_index(drop = True).copy().shape[0]
		pct_dropouts_avg = num_dropouts_avg/num_points * 100

		#Number of points greater than their mode - NOT MASTER AVERAGE, EACH AIRCRAFT HAS THEIR OWN
		num_dropouts_mode = points_master[(points_master['dropout_length'] > points_master['points_mode'])].reset_index(drop = True).copy().shape[0]
		pct_dropouts_mode = num_dropouts_mode/num_points * 100

		'''
		#Print calculated data
		print("Points: " + str(num_points))
		print(">Points Average: " + str(points_master['points_avg'][0]))
		print(">Points Mode: " + str(points_master['dropout_length'].mode()[0]))
		print("Dropouts [Points > Average]: " + str(num_dropouts_avg))
		print(">Percentage: {:.4f}%".format(pct_dropouts_avg))

		print("Dropouts [Points > Mode]: " + str(num_dropouts_mode))
		print(">Percentage: {:.4f}%".format(pct_dropouts_mode))
		'''


		#Standard Deviation range value counts
		abv_avg = points_master.copy()
		below_zero_dev = abv_avg[(abv_avg['points_zscore'] <= 0)].reset_index(drop = True).copy().shape[0]
		one_dev = abv_avg[(abv_avg['points_zscore'] > 0) & (abv_avg['points_zscore'] <= 1)].reset_index(drop = True).copy().shape[0]
		two_dev = abv_avg[(abv_avg['points_zscore'] > 1) & (abv_avg['points_zscore'] <= 2)].reset_index(drop = True).copy().shape[0]
		three_dev = abv_avg[(abv_avg['points_zscore'] > 2) & (abv_avg['points_zscore'] <= 3)].reset_index(drop = True).copy().shape[0]
		over_three_dev = abv_avg[(abv_avg['points_zscore'] > 3)].reset_index(drop = True).copy().shape[0]

		'''
		#Print Standard Deviation value counts summary
		print("\nStandard Deviation Counts:")
		print("[TOTAL]: " + str(one_dev + two_dev + three_dev + over_three_dev))
		print("[<=  0]: " + str(below_zero_dev))
		print("[<= +1]: " + str(one_dev))
		print("[<= +2]: " + str(two_dev))
		print("[<= +3]: " + str(three_dev))
		print("[>  +3]: " + str(over_three_dev))
		'''


		#Calculate Holistic Z-Score column
		points_master['holistic_zscore'] = (points_master.dropout_length - points_master.dropout_length.mean())/points_master.dropout_length.std(ddof=0)

		#Add information to metadata
		meta = dict({
			'num_craft': points_master['icao24'].unique().shape[0],
			'num_points': points_master['dropout_length'].shape[0],
			'avg': points_master['dropout_length'].mean(),
			'mode': points_master['dropout_length'].mode(),
			'std_dev': points_master['dropout_length'].std(),
			'min': points_master['dropout_length'].min(),
			'max': points_master['dropout_length'].max()
			})
		meta = pd.DataFrame(meta)
		metadata = metadata.append(meta, ignore_index = True)


		# sns.set(rc=
			# {
			# 	'axes.facecolor': 'white',
			# 	'figure.facecolor':'gray',
			# 	'grid.color': 'white',
			# 	'xtick.color': 'white',
			# 	'ytick.color': 'white',
			# 	'axes.labelcolor': 'black',
			# 	'text.color': 'black',
			# 	'axes.edgecolor': 'white',
			# 	"xtick.labelsize": 12,
			# 	"ytick.labelsize": 12,
			# })
		#sns.set_style('darkgrid')
		
		
		
		######################################################################################################################################################
		# PLOT CONTROLS																																		 #
		######################################################################################################################################################
		#Category Description Plots
		
		group_mean_vs_mode_loli = False
		dropout_length_grouped_barh = False
		hol_zscore_violin_category = False
		hol_zscore_category_foc = False
		hol_zscore_category = False
		
		
		
		
		#Uncategorized Plots
		hol_pct_drop_pie = False
		pct_drop_pie = False
		hol_zscore_kde = False
		hol_zscore_kde_foc = False
		hol_zscore_ecdf_foc = False
		hol_zscore_ecdf = False
		hol_zscore_hist_log = False
		clustering_plots = False
		z_score_joint_velocity_foc = False
		z_score_joint_geoaltitude_foc = False
		z_score_distribution_pie = False
		z_score_dist_kde = False
		dropout_length_dist_minmax = False
		dropout_length_dist_focused = False
		######################################################################################################################################################
		
		
		
		#Create group data and add mode column
		groups = points_master.groupby(by = "categoryDescription")['dropout_length'].describe().reset_index()
		temp = points_master.groupby(by = "categoryDescription")['dropout_length'].agg(lambda x:x.value_counts().index[0])
		print(temp)
		groups.insert(groups.shape[1], 'mode', temp.values)
		print(groups)
		
		
		#DO BAR OF PIE NEXT: https://matplotlib.org/stable/gallery/pie_and_polar_charts/bar_of_pie.html#sphx-glr-gallery-pie-and-polar-charts-bar-of-pie-py
		
		
		
	
		if(group_mean_vs_mode_loli == True):
			
			fig, ax = plt.subplots(figsize = (10, 6.5))
			
			#Create group data and add mode column
			groups = points_master.groupby(by = "categoryDescription")['dropout_length'].describe().reset_index()
			temp = points_master.groupby(by = "categoryDescription")['dropout_length'].agg(lambda x:x.value_counts().index[0])
			print(temp)
			groups.insert(groups.shape[1], 'mode', temp.values)
			print(groups)

			
			# Reorder it following the values of the first value:
			ordered_df = groups.sort_values(by='mean')
			my_range=range(1,len(groups.index)+1)
			
			# The horizontal plot is made using the hline function
			plt.hlines(y=my_range, xmin=ordered_df['mode'], xmax=ordered_df['mean'], color='#AEAEAE', alpha=1, zorder = 1)
			#plt.hlines(y=my_range, xmin=ordered_df['mean'], xmax=ordered_df['mode'], color='grey', alpha=0.4)
			
			plt.scatter(ordered_df['mean'], my_range, color='#FF671F', alpha=1, label='mean', zorder = 2)
			plt.scatter(ordered_df['mode'], my_range, color='#009A44', alpha=1 , label='mode', zorder = 3)
			plt.legend()
			
			# Add title and axis names
			plt.yticks(my_range, ordered_df['categoryDescription'])
			plt.suptitle("Why Mean is a bad detection metric", weight = 'bold', fontsize = 14)
			plt.title("Comparison of the Mean and the Mode for each Group")
			plt.xlabel('Value of the variables')
			plt.ylabel('Group')

			# Show the graph
			plt.tight_layout()
			plt.show()


		
		
		
		if(dropout_length_grouped_barh == True):
			height = list(groups['mean'])
			#print(height)
			bars = list(groups['categoryDescription'])
			#print(bars)
			y_pos = np.arange(len(bars))
			print(y_pos)
			
			# Create bars
			plt.barh(y_pos, height)

			# Create names on the x-axis
			plt.yticks(y_pos, bars)

			# Show graphic
			plt.tight_layout()
			plt.show()
			
			#ax = points_master.groupby(by = "categoryDescription")['dropout_length'].mean().plot(kind = "bar")
			#plt.show()
		
		
		if (hol_zscore_violin_category == True):
			sns.violinplot(data = points_master, x = 'points_zscore', y = 'categoryDescription')
			plt.show()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		######################################################################################################################################################
		# HOLISTIC Z-SCORE DISTRIBUTION BY CATEGORY DESCRIPTION	PLOTS																						 #
		######################################################################################################################################################
		#Holistic Z-Score Distribution by Category Description (Focused)
		if(hol_zscore_category_foc == True):
			
			sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), "figure.figsize":(10, 6.5)})
			#sns.set(rc={"figure.figsize":(10, 6.5)})
			category_names = list(points_master['categoryDescription'].unique())
			# we generate a color palette with Seaborn.color_palette()
			pal = sns.color_palette(palette='bright', n_colors=len(category_names))
			
			# sns.kdeplot(points_master['holistic_zscore'], hue = points_master['categoryDescription'])
			# plt.show()
			
			
			# in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
			g = sns.FacetGrid(points_master, row = "categoryDescription", hue = 'categoryDescription', aspect=10, height=6.5/len(category_names), palette=pal)
			
			# then we add the densities kdeplots for each month
			g.map(sns.kdeplot, 'holistic_zscore',
				bw_adjust=1.2, clip_on=False,
				fill=True, alpha=1, linewidth=1.5)
				
			# here we add a white line that represents the contour of each kdeplot
			g.map(sns.kdeplot, 'holistic_zscore', 
				bw_adjust=1.2, clip_on=False, 
				color="w", lw=2)
			
			# here we add a horizontal line for each plot
			g.map(plt.axhline, y=0,
				lw=2, clip_on=False)

			
			# we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
			# notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
			for i, ax in enumerate(g.axes.flat):
				
				ax.set_xlim(-1, 1)
				ax.text(ax.get_xlim()[0], -0.35, category_names[i],
						fontweight='bold', fontsize=10,
						#color = 'black')
						color=ax.lines[-1].get_color())
				
				ax.tick_params(color = 'white', pad = 10)
				ax.set_ylabel(ax.get_ylabel(), color = 'white')
			
			# we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
			g.fig.subplots_adjust(hspace=-0.3)
			
			# eventually we remove axes titles, yticks and spines
			g.set_titles("")
			g.set(yticks=[])
			g.despine(bottom=True, left=True)

			plt.setp(ax.get_xticklabels(), fontsize=10, fontweight='bold')
			plt.xlabel('Z-Score', fontweight='bold', fontsize=10)
			g.fig.suptitle('Holistic Z-Score Distribution by Aircraft Category (Focused)\nNumber of Aircraft: {}'.format(num_craft),
						fontsize=12,
						fontweight='bold')
			#plt.tight_layout()
			
			
			plt.show()
			fig = g.fig
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
			
			
			
			""" import joypy
			fig, ax = joypy.joyplot(points_master, by = "categoryDescription", column = "holistic_zscore", linecolor = 'white', figsize = (10,6.5), x_range = (-1, 1), colormap = cm.autumn)
			
			
			for i in range(len(points_master['categoryDescription'].unique())):
				y_position = ax[i].get_ylim()[1] / 3.5  # adjust with ylim for each plot
				ax[i].text(9, y_position, points_master['categoryDescription'].unique()[i], color = "red")
			# for a in ax[:-1]:
			# 	a.set_yticklabels("Helo")
			# month_dict = list(points_master['categoryDescription'])
			#  for i, ax in enumerate(ax[:-1]):
			# 	ax.text(-15, 0.02, month_dict[i+1],
			# 			fontweight='bold', fontsize=15,
			# 			color=ax.lines[-1].get_color())
			
			
			
			# for a in ax[:-1]:
			# 	a.set_xticklabels(get_xticklabels(), fontsize=15, fontweight='bold')
			# 	# 	a.set_xlim([-3,3])
			# 	# 	a.set_xticklabels('points_zscore')
			
			
			#plt.show() """
		
		
		#Holistic Z-Score Distribution by Category Description
		if(hol_zscore_category == True):
			
			sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), "figure.figsize":(10, 6.5)})
			#sns.set(rc={"figure.figsize":(10, 6.5)})
			category_names = list(points_master['categoryDescription'].unique())
			# we generate a color palette with Seaborn.color_palette()
			pal = sns.color_palette(palette='bright', n_colors=len(category_names))
			
			# sns.kdeplot(points_master['holistic_zscore'], hue = points_master['categoryDescription'])
			# plt.show()
			
			
			# in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
			g = sns.FacetGrid(points_master, row = "categoryDescription", hue = 'categoryDescription', aspect=10, height=6.5/len(category_names), palette=pal)
			
			# then we add the densities kdeplots for each month
			g.map(sns.kdeplot, 'holistic_zscore',
				bw_adjust=1.2, clip_on=False,
				fill=True, alpha=1, linewidth=1.5)
				
			# here we add a white line that represents the contour of each kdeplot
			g.map(sns.kdeplot, 'holistic_zscore', 
				bw_adjust=1.2, clip_on=False, 
				color="w", lw=2)
			
			# here we add a horizontal line for each plot
			g.map(plt.axhline, y=0,
				lw=2, clip_on=False)

			
			# we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
			# notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
			for i, ax in enumerate(g.axes.flat):
				
				#ax.set_xlim(-1, 1)
				ax.text(ax.get_xlim()[0], -0.35, category_names[i],
						fontweight='bold', fontsize=10,
						#color = 'black')
						color=ax.lines[-1].get_color())
				
				ax.tick_params(color = 'white', pad = 10)
				ax.set_ylabel(ax.get_ylabel(), color = 'white')
			
			# we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
			g.fig.subplots_adjust(hspace=-0.3)
			
			# eventually we remove axes titles, yticks and spines
			g.set_titles("")
			g.set(yticks=[])
			g.despine(bottom=True, left=True)

			plt.setp(ax.get_xticklabels(), fontsize=10, fontweight='bold')
			plt.xlabel('Z-Score', fontweight='bold', fontsize=10)
			g.fig.suptitle('Holistic Z-Score Distribution by Aircraft Category (Focused)\nNumber of Aircraft: {}'.format(num_craft),
						fontsize=12,
						fontweight='bold')
			#plt.tight_layout()
			
			
			plt.show()
			fig = g.fig
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
		######################################################################################################################################################
		
		
		
		######################################################################################################################################################
		# DROPOUT PERCENTAGE PIE CHARTS																														 #
		######################################################################################################################################################
		#Holistic Percentage Dropouts Pie Charts - Average and Mode
		if(hol_pct_drop_pie == True):

			#Create figure
			fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 6.5))
			fig.set_facecolor("white")
			
			#Labels
			ax1_labels = ["Dropout", "Normal"]
			ax2_labels = ["Dropout", "Normal"]
			
			#Data
			ax1_data = [
				points_master[points_master['dropout_length'] > points_master['dropout_length'].mean()].reset_index(drop = True).copy().shape[0],
				points_master[points_master['dropout_length'] <= points_master['dropout_length'].mean()].reset_index(drop = True).copy().shape[0]
				]
			ax2_data = [
				points_master[points_master['dropout_length'] > points_master['dropout_length'].mode()[0]].reset_index(drop = True).copy().shape[0],
				points_master[points_master['dropout_length'] <= points_master['dropout_length'].mode()[0]].reset_index(drop = True).copy().shape[0]]
			
			#Autopct
			def make_autopct(values):
				def my_autopct(pct):
					total = sum(values)
					val = int(round(pct*total/100.0))
					return '{p: .2f}%\n({v:d})'.format(p=pct,v=val)
				return my_autopct
			
			#Plot
			ax1_wedges, ax1_texts, ax1_autotexts = ax1.pie(ax1_data, labels = ax1_labels, labeldistance = None, autopct = make_autopct(ax1_data), wedgeprops = {'width' :0.4, 'linewidth' : 3, 'edgecolor': '#AEAEAE'}, pctdistance = 0.8, explode=[0.00]*len(ax1_data), colors = ['#000', '#009A44'])
			ax2_wedges, ax2_texts, ax2_autotexts = ax2.pie(ax2_data, labels = ax2_labels, labeldistance = None, autopct = make_autopct(ax1_data), wedgeprops = {'width' :0.4, 'linewidth' : 3, 'edgecolor': '#AEAEAE'}, pctdistance = 0.8, explode=[0.00]*len(ax1_data), colors = ['#000', '#009A44'])
			
			#Set text color
			plt.setp(ax1_autotexts, color = 'white', size=10, weight="bold")
			plt.setp(ax2_autotexts, color = 'white', size=10, weight="bold")

			#Add patch to show values in the center of the circle
			import matplotlib.patheffects as path_effects
			ax1.text(0, 0, "Average:\n{:.2f}".format(points_master['dropout_length'].mean()), ha='center', va='center', fontsize=28, weight = 'normal', color = 'black')#.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
			ax2.text(0, 0, "Mode:\n{:.2f}".format(points_master['dropout_length'].mode()[0]), ha='center', va='center', fontsize=28, weight = 'normal', color = 'black')#.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
			

			#Legend color patches
			import matplotlib.patches as mpatches
			drop_patch = mpatches.Patch(color='black', label='Dropout')
			norm_patch = mpatches.Patch(color='#009A44', label='Normal')

			#Legend
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=False, ncol=2, handles = [drop_patch, norm_patch])
			ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=False, ncol=2, handles = [drop_patch, norm_patch])

			#Titles
			ax1.set_title("Points > Average", weight = 'bold', fontsize = 14)
			ax2.set_title("Points > Mode", weight = 'bold', fontsize = 14)
			plt.suptitle("Holistic Percentage of Dropouts for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tAverage: " + str("N/A") + "\tDropouts: " + str(num_dropouts_avg) + "\tPercent Dropouts: {:0.2f}%".format(pct_dropouts_avg)).expandtabs()
			#title += "\n[Ranges are inclusive of the outside value, ex. +2 would be is > 2 to 3"
			#plt.title(title)
			
			#Display options
			plt.tight_layout()
			#plt.show()
			pdf.savefig(fig)
			plt.close("all")
		
		
		#Percentage Dropouts Pie Charts - Average and Mode
		if(pct_drop_pie == True):

			#Create figure
			fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 6.5))
			fig.set_facecolor("white")
			
			#Labels
			ax1_labels = ["Dropout", "Normal"]
			ax2_labels = ["Dropout", "Normal"]
			
			#Data
			ax1_data = [
				points_master[points_master['dropout_length'] > points_master['points_avg']].reset_index(drop = True).copy().shape[0],
				points_master[points_master['dropout_length'] <= points_master['points_avg']].reset_index(drop = True).copy().shape[0]
				]
			ax2_data = [
				points_master[points_master['dropout_length'] > points_master['points_mode']].reset_index(drop = True).copy().shape[0],
				points_master[points_master['dropout_length'] <= points_master['points_mode']].reset_index(drop = True).copy().shape[0]]
			
			#Autopct
			def make_autopct(values):
				def my_autopct(pct):
					total = sum(values)
					val = int(round(pct*total/100.0))
					return '{p: .2f}%\n({v:d})'.format(p=pct,v=val)
				return my_autopct
			
			#Plot
			ax1_wedges, ax1_texts, ax1_autotexts = ax1.pie(ax1_data, labels = ax1_labels, labeldistance = None, autopct = make_autopct(ax1_data), wedgeprops = {'width' :0.4, 'linewidth' : 3, 'edgecolor': '#AEAEAE'}, pctdistance = 0.8, explode=[0.00]*len(ax1_data), colors = ['#000', '#009A44'])
			ax2_wedges, ax2_texts, ax2_autotexts = ax2.pie(ax2_data, labels = ax2_labels, labeldistance = None, autopct = make_autopct(ax1_data), wedgeprops = {'width' :0.4, 'linewidth' : 3, 'edgecolor': '#AEAEAE'}, pctdistance = 0.8, explode=[0.00]*len(ax1_data), colors = ['#000', '#009A44'])
			
			#Set text color
			plt.setp(ax1_autotexts, color = 'white', size=10, weight="bold")
			plt.setp(ax2_autotexts, color = 'white', size=10, weight="bold")

			# #Add patch to show values in the center of the circle
			# import matplotlib.patheffects as path_effects
			# ax1.text(0, 0, "Average:\n{:.2f}".format(points_master['dropout_length'].mean()), ha='center', va='center', fontsize=28, weight = 'normal', color = 'black')#.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
			# ax2.text(0, 0, "Mode:\n{:.2f}".format(points_master['dropout_length'].mode()[0]), ha='center', va='center', fontsize=28, weight = 'normal', color = 'black')#.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
			

			#Legend color patches
			import matplotlib.patches as mpatches
			drop_patch = mpatches.Patch(color='black', label='Dropout')
			norm_patch = mpatches.Patch(color='#009A44', label='Normal')

			#Legend
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=False, ncol=2, handles = [drop_patch, norm_patch])
			ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=False, ncol=2, handles = [drop_patch, norm_patch])

			#Titles
			ax1.set_title("Points > Average", weight = 'bold', fontsize = 14)
			ax2.set_title("Points > Mode", weight = 'bold', fontsize = 14)
			plt.suptitle("Percentage of Dropouts for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tAverage: " + str("N/A") + "\tDropouts: " + str(num_dropouts_avg) + "\tPercent Dropouts: {:0.2f}%".format(pct_dropouts_avg)).expandtabs()
			#title += "\n[Ranges are inclusive of the outside value, ex. +2 would be is > 2 to 3"
			#plt.title(title)
			
			#Display options
			plt.tight_layout()
			#plt.show()
			pdf.savefig(fig)
			plt.close("all")
			
			
			'''
			#ORIGINAL CIRCLES BEFORE HOLISTIC WAS COPIED AND SWAPPED TO NON-HOLISTIC DATA
			fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 6.5))
			
			#Labels
			ax1_labels = ["Dropout", "Normal"]
			ax2_labels = ["Dropout", "Normal"]

			#Data
			ax1_data = [
				points_master[points_master['dropout_length'] > points_master['points_avg']].reset_index(drop = True).copy().shape[0],
				points_master[points_master['dropout_length'] <= points_master['points_avg']].reset_index(drop = True).copy().shape[0]
				]
			ax2_data = [
				points_master[points_master['dropout_length'] > points_master['points_mode']].reset_index(drop = True).copy().shape[0],
				points_master[points_master['dropout_length'] <= points_master['points_mode']].reset_index(drop = True).copy().shape[0]]
			
			#Autopct
			def make_autopct(values):
				def my_autopct(pct):
					total = sum(values)
					val = int(round(pct*total/100.0))
					return '{p: .2f}%\n({v:d})'.format(p=pct,v=val)
				return my_autopct
			
			# Create a circle at the center of the plot
			from matplotlib.patches import Circle
			# ax1_circle = Circle( (0,0), 0.7, color='white')
			# ax2_circle = Circle( (0,0), 0.7, color='white')
			# ax1.add_patch(ax1_circle)
			# ax2.add_patch(ax2_circle)
			import matplotlib.patches as mpatches
			drop_patch = mpatches.Patch(color='black', label='Dropout')
			norm_patch = mpatches.Patch(color='#009A44', label='Normal')
			

			#Plot
			ax1_wedges, ax1_texts, ax1_autotexts = ax1.pie(ax1_data, labels = ax1_labels, labeldistance = None, autopct = make_autopct(ax1_data), wedgeprops = {'width' :0.4, 'linewidth' : 2, 'edgecolor': 'white'}, pctdistance = 0.8, explode=[0.00]*len(ax1_data), colors = ['#000', '#009A44'])
			ax2_wedges, ax2_texts, ax2_autotexts = ax2.pie(ax2_data, labels = ax2_labels, labeldistance = None, autopct = make_autopct(ax1_data), wedgeprops = {'width' :0.4, 'linewidth' : 2, 'edgecolor': 'white'}, pctdistance = 0.8, explode=[0.00]*len(ax1_data), colors = ['#000', '#009A44'])
			
			plt.setp(ax1_autotexts, color = 'white', size=10, weight="bold")
			plt.setp(ax2_autotexts, color = 'white', size=10, weight="bold")
			#ax1.legend(loc = 'lower center')
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=False, ncol=2, handles = [drop_patch, norm_patch])
			ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=False, ncol=2, handles = [drop_patch, norm_patch])
			
			#ax1.set_frame_on(True)
			#ax1.set_facecolor("#AEAEAE")
			#plt.setp(ax1.spines.values(),visible=False)
			
			ax1.set_title("Points > Average", weight = 'bold', fontsize = 14)
			ax2.set_title("Points > Mode", weight = 'bold', fontsize = 14)

			fig.set_facecolor("#AEAEAE")

			plt.suptitle("Percentage of Dropouts for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tAverage: " + str("N/A") + "\tDropouts: " + str(num_dropouts_avg) + "\tPercent Dropouts: {:0.2f}%".format(pct_dropouts_avg)).expandtabs()
			#title += "\n[Ranges are inclusive of the outside value, ex. +2 would be is > 2 to 3"
			#plt.title(title)
			plt.tight_layout()
			plt.show()
			'''
		######################################################################################################################################################
		
		
		
		######################################################################################################################################################
		# KERNEL DENSITY ESTIMATE Z-SCORE DISTRIBUTION PLOTS																								 #
		######################################################################################################################################################
		#Holistic Z-Score Distribution Plot - Kernel Density Estimates
		if(hol_zscore_kde == True):

			#Create plot
			ax = sns.displot(points_master, x = "holistic_zscore", kind = "kde", hue = "icao24", fill = True, legend = False, alpha=0.6)
			#ax.set(xlim=(-1, 1))
			ax.set(xlim=(points_master['holistic_zscore'].min()-1, points_master['holistic_zscore'].max()+1))

			#Legend
			desc = points_master['holistic_zscore'].describe()
			desc = desc.apply(lambda x: format(x, '.4f'))
			desc['count'] = int(pd.to_numeric(desc['count']))
			#print(desc)
			data1=[i for i in desc.index]
			data2=[str(i) for i in desc]
			text= ('\n'.join([ a +':'+ b for a,b in zip(data1,data2)]))
			leg = plt.legend(labels = [text], title = "Point Info", loc = 'best', bbox_to_anchor = (1,1), title_fontproperties={'weight':'bold', 'size':12}, handlelength=0, handletextpad=0, edgecolor = 'black')#, fancybox=True)
			for item in leg.legendHandles:
				item.set_visible(False)
			for t in leg.get_texts():
				t.set_ha('left')
			
			#Figure size
			ax.fig.set_figwidth(10)
			ax.fig.set_figheight(6.5)
			
			#Labels
			ax.set_xlabels("Z-Score")
			
			#Titles
			plt.suptitle("Holistic Z-Score Kernel Density Estimates Distribution", weight = 'bold').set_fontsize('16')
			title = ("Number of Aircraft: " + str(num_craft))# + "\tData Points: " + str(num_points)).expandtabs()# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			plt.title(title, weight='bold', fontsize = 14)
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			fig = ax.ax.get_figure()
			pdf.savefig(fig)
			plt.close("all")
		
		
		#Holistic Z-Score Distribution Plot - Kernel Density Estimates (Focused)
		if(hol_zscore_kde_foc == True):
			#Create plot
			ax = sns.displot(points_master, x = "holistic_zscore", kind = "kde", hue = "icao24", fill = True, legend = False, alpha=0.6)
			ax.set(xlim=(-1, 1))
			#ax.set(xlim=(points_master['holistic_zscore'].min()-1, points_master['holistic_zscore'].max()+1))

			#Legend
			desc = points_master['holistic_zscore'].describe()
			desc = desc.apply(lambda x: format(x, '.4f'))
			desc['count'] = int(pd.to_numeric(desc['count']))
			#print(desc)
			data1=[i for i in desc.index]
			data2=[str(i) for i in desc]
			text= ('\n'.join([ a +':'+ b for a,b in zip(data1,data2)]))
			leg = plt.legend(labels = [text], title = "Point Info", loc = 'best', bbox_to_anchor = (1,1), title_fontproperties={'weight':'bold', 'size':12}, handlelength=0, handletextpad=0, edgecolor = 'black')#, fancybox=True)
			for item in leg.legendHandles:
				item.set_visible(False)
			for t in leg.get_texts():
				t.set_ha('left')
			
			#Figure size
			ax.fig.set_figwidth(10)
			ax.fig.set_figheight(6.5)
			
			#Labels
			ax.set_xlabels("Z-Score")
			
			#Titles
			plt.suptitle("Holistic Z-Score Kernel Density Estimates Distribution (Focused)", weight = 'bold').set_fontsize('16')
			title = ("Number of Aircraft: " + str(num_craft))# + "\tData Points: " + str(num_points)).expandtabs()# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			plt.title(title, weight='bold', fontsize = 14)
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			fig = ax.ax.get_figure()
			pdf.savefig(fig)
			plt.close("all")
		######################################################################################################################################################
		
		
		
		######################################################################################################################################################
		# EMPIRICAL CUMULATIVE Z-SCORE DISTRIBUTION PLOTS																									 #
		######################################################################################################################################################
		#Holistic Z-Score Distribution Plot - Empirical Cumulative (Focused)
		if(hol_zscore_ecdf_foc == True):

			#Style options
			sns.set(rc=
			{
				'axes.facecolor': 'white',
			 	'figure.facecolor':'white',
			 	'grid.color': 'white',
			 	'xtick.color': 'black',
			 	'ytick.color': 'black',
			# 	'axes.labelcolor': 'black',
			# 	'text.color': 'black',
			 	'axes.edgecolor': 'black',
			# 	"xtick.labelsize": 12,
			# 	"ytick.labelsize": 12,
			#	'figure.figsize': (10, 6.5),
			})
			
			#Create Plot
			ax = sns.displot(points_master, x = "holistic_zscore", kind = "ecdf", color = 'black')#, hue = "icao24", fill = True, legend = False)
			
			#Set limits
			ax.set(xlim=(-1, 1))
			#ax.set(xlim=(points_master['holistic_zscore'].min()-1, points_master['holistic_zscore'].max()+1))

			#Shading
			x1 = ax.ax.get_lines()[0].get_xydata()[:,0]
			y1 = ax.ax.get_lines()[0].get_xydata()[:,1]
			plt.axvspan(-1, 0, color='#009A44', alpha=0.8, lw=0)
			plt.axvspan(0, 1, color='#FF671F', alpha=0.8, lw=0)
			ax.ax.fill_between(x1,y1, np.max(y1), color="white", alpha=1)
			
			for tick in ax.ax.get_yticks():
				ax.ax.axhline(tick,color = '#AEAEAE',linestyle='dashed',lw=1)
			
			#Legend
			desc = points_master['holistic_zscore'].describe()
			desc = desc.apply(lambda x: format(x, '.4f'))
			desc['count'] = int(pd.to_numeric(desc['count']))
			#print(desc)
			data1=[i for i in desc.index]
			data2=[str(i) for i in desc]
			text= ('\n'.join([ a +':'+ b for a,b in zip(data1,data2)]))
			leg = plt.legend(labels = [text], title = "Point Info", loc = 'best', bbox_to_anchor = (1,1), title_fontproperties={'weight':'bold', 'size':10}, handlelength=0, handletextpad=0, edgecolor = 'black')
			for item in leg.legendHandles:
				item.set_visible(False)
			for t in leg.get_texts():
				t.set_ha('left')
			
			
			#Figure size
			ax.fig.set_figwidth(10)
			ax.fig.set_figheight(6.5)

			#Labels
			ax.set_xlabels("Z-Score", weight = 'bold', fontsize = '12')
			ax.set_ylabels("% of Z-Scores Below Value", weight = 'bold', fontsize = '12')
			
			#Titles
			plt.suptitle("Holistic Z-Score Empirical Cumulative Distribution (Focused)", weight = 'bold').set_fontsize('16')
			title = ("Number of Aircraft: " + str(num_craft))# + "\tData Points: " + str(num_points)).expandtabs()# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			plt.title(title, weight = 'bold', fontsize = '14')
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			fig = ax.ax.get_figure()
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
		

		#Holistic Z-Score Distribution Plot - Empirical Cumulative
		if(hol_zscore_ecdf == True):
				
				
			#Style options
			sns.set(rc=
			{
				'axes.facecolor': 'white',
			 	'figure.facecolor':'white',
			 	'grid.color': 'white',
			 	'xtick.color': 'black',
			 	'ytick.color': 'black',
			# 	'axes.labelcolor': 'black',
			# 	'text.color': 'black',
			 	'axes.edgecolor': 'black',
			# 	"xtick.labelsize": 12,
			# 	"ytick.labelsize": 12,
			#	'figure.figsize': (10, 6.5),
			})
			
			#Create Plot
			ax = sns.displot(points_master, x = "holistic_zscore", kind = "ecdf", color = 'black')#, hue = "icao24", fill = True, legend = False)
			
			#Set limits
			#ax.set(xlim=(-1, 1))
			ax.set(xlim=(points_master['holistic_zscore'].min()-1, points_master['holistic_zscore'].max()+1))

			#Shading
			x1 = ax.ax.get_lines()[0].get_xydata()[:,0]
			y1 = ax.ax.get_lines()[0].get_xydata()[:,1]
			plt.axvspan(-1, 0, color='#009A44', alpha=0.8, lw=0)
			plt.axvspan(0, 1, color='#FF671F', alpha=0.8, lw=0)
			ax.ax.fill_between(x1,y1, np.max(y1), color="white", alpha=1)
			
			for tick in ax.ax.get_yticks():
				ax.ax.axhline(tick,color = '#AEAEAE',linestyle='dashed',lw=1)
			
			#Legend
			desc = points_master['holistic_zscore'].describe()
			desc = desc.apply(lambda x: format(x, '.4f'))
			desc['count'] = int(pd.to_numeric(desc['count']))
			#print(desc)
			data1=[i for i in desc.index]
			data2=[str(i) for i in desc]
			text= ('\n'.join([ a +':'+ b for a,b in zip(data1,data2)]))
			leg = plt.legend(labels = [text], title = "Point Info", loc = 'best', bbox_to_anchor = (1,1), title_fontproperties={'weight':'bold', 'size':10}, handlelength=0, handletextpad=0, edgecolor = 'black')
			for item in leg.legendHandles:
				item.set_visible(False)
			for t in leg.get_texts():
				t.set_ha('left')
			
			
			#Figure size
			ax.fig.set_figwidth(10)
			ax.fig.set_figheight(6.5)

			#Labels
			ax.set_xlabels("Z-Score", weight = 'bold', fontsize = '12')
			ax.set_ylabels("% of Z-Scores Below Value", weight = 'bold', fontsize = '12')
			
			#Titles
			plt.suptitle("Holistic Z-Score Empirical Cumulative Distribution", weight = 'bold').set_fontsize('16')
			title = ("Number of Aircraft: " + str(num_craft))# + "\tData Points: " + str(num_points)).expandtabs()# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			plt.title(title, weight = 'bold', fontsize = '14')
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			fig = ax.ax.get_figure()
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
		######################################################################################################################################################

		
		######################################################################################################################################################
		# HISTOGRAM Z-SCORE DISTRIBUTION PLOT																												 # 
		######################################################################################################################################################
		#Holistic Z-Score Distribution Plot - Histogram (Log Scale)
		if(hol_zscore_hist_log == True):
			
			#Style
			sns.set(rc=
				{
					'axes.facecolor': 'white',
					'figure.facecolor':'white',
					'grid.color': '#AEAEAE',
					'xtick.color': 'black',
					'ytick.color': 'black',
					'axes.labelcolor': 'black',
					'text.color': 'black',
					'axes.edgecolor': 'black',
					'xtick.labelsize': 12,
					'ytick.labelsize': 12,
					'legend.title_fontsize': 12,
				})
			import matplotlib.colors as colors
			pal = sns.diverging_palette(146.49, 0, as_cmap=True)
			bounds = np.array([-5, 0, 5, 10, 15, 20, 25])
			
			#Conditionally show Z-Score Color Legend
			if (q <= 10):
				ax = sns.displot(points_master, x = "holistic_zscore", kind = "hist", bins = 50, log_scale=(False, True), hue = "holistic_zscore", palette='RdYlGn_r', alpha = 1, hue_norm=mpl.colors.CenteredNorm(), edgecolor='black', legend = True)#hue_norm = colors.BoundaryNorm(boundaries=bounds, ncolors=666))# #, hue = "icao24", fill = True, legend = False)
				
				#Z-Score Color Legend
				handles, labels = ax.ax.get_legend_handles_labels()
				new_labels = ["{:.3f}".format(label) for label in labels]
				ax.ax.legend(handles, new_labels, title = 'hr', loc = 'best')
				for text in ax.legend.texts:
					text.set_text("{:0.3f}".format(float(text.get_text())))
				ax.legend.set(title="Z-Score", bbox_to_anchor = (0.98,0.4))#, frameon = True)
			else:
				ax = sns.displot(points_master, x = "holistic_zscore", kind = "hist", bins = 50, log_scale=(False, True), hue = "holistic_zscore", palette='RdYlGn_r', alpha = 1, hue_norm=mpl.colors.CenteredNorm(), edgecolor='black', legend = False)#hue_norm = colors.BoundaryNorm(boundaries=bounds, ncolors=666))# #, hue = "icao24", fill = True, legend = False)
			
			#Axis limits
			ax.set(xlim=(points_master['holistic_zscore'].min()-1, points_master['holistic_zscore'].max()+1))
			#ax.set(xlim=(-3.5, 3.5))

			#Point info
			desc = points_master['holistic_zscore'].describe()
			desc = desc.apply(lambda x: format(x, '.4f'))
			desc['count'] = int(pd.to_numeric(desc['count']))
			#print(desc)
			data1=[i for i in desc.index]
			data2=[str(i) for i in desc]
			text= ('\n'.join([ a +':'+ b for a,b in zip(data1,data2)]))
			leg = plt.legend(labels = [text], title = "Point Info", loc = 'best', bbox_to_anchor = (1,1), title_fontproperties={'weight':'bold', 'size':10}, handlelength=0, handletextpad=0, edgecolor = 'black')
			for item in leg.legendHandles:
				item.set_visible(False)
			for t in leg.get_texts():
				t.set_ha('left')
			
			#Figure size
			ax.fig.set_figwidth(10)
			ax.fig.set_figheight(6.5)

			#Labels & Titles
			ax.set_xlabels("Z-Score", weight = 'bold', fontsize = 12)
			ax.set_ylabels("Count (Log Scale)", weight = 'bold', fontsize = 12)
			plt.suptitle("Holistic Z-Score Distribution (Log Scale) for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			#title = ("Number of Aircraft: " + str(num_craft) + "\tData Points: " + str(num_points)).expandtabs()# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			#plt.title(title)
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			fig = ax.ax.get_figure()
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
		######################################################################################################################################################
		
		
		
		######################################################################################################################################################
		# SKLEARN KMEANS & DBSCAN CLUSTERING PLOTS																											 
		######################################################################################################################################################
		#Sklearn Clustering Plots
		if(clustering_plots == True):
				

			from sklearn.cluster import DBSCAN
			from sklearn import metrics
			from sklearn.datasets import make_blobs
			from sklearn.preprocessing import StandardScaler

			fig, ax = plt.subplots()
			#X = np.array(points_master.loc(axis=0)[:, :, 'points_zscore', 'velocity'].dropna())
			X = points_master[['points_zscore', 'velocity']].dropna().to_numpy()

			# cluster the data into five clusters
			from sklearn.cluster import KMeans
			kmeans = KMeans(n_clusters=5)
			kmeans.fit(X)
			y_pred = kmeans.predict(X)# plot the cluster assignments and cluster centers
			ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="plasma")
			ax.scatter(kmeans.cluster_centers_[:, 0],   
						kmeans.cluster_centers_[:, 1],
						marker='^', 
						c=[0, 1, 2, 3, 4], 
						s=100, 
						linewidth=2,
						cmap="plasma")
			plt.xlabel("Point Z-Score")
			plt.ylabel("Velocity (m/s)")
			plt.suptitle("K-Means Clustering: Z-Score & Velocity", weight = 'bold').set_fontsize('16')
			plt.title("Number of Aircraft: " + str(num_craft))
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()










			scaler = StandardScaler()
			X_scaled = scaler.fit_transform(X)

			dbscan = DBSCAN(eps = 0.123, min_samples = 2)
			clusters = dbscan.fit_predict(X_scaled)
			# plot the cluster assignments
			fig, ax = plt.subplots()
			ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap="plasma")
			plt.xlabel("Point Z-Score")
			plt.ylabel("Velocity (m/s)")
			plt.suptitle("DBSCAN Clustering: Z-Score & Velocity", weight = 'bold').set_fontsize('16')
			plt.title("Number of Aircraft: " + str(num_craft))
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
			#clustering.labels_

			# from sklearn.metrics.cluster import adjusted_rand_score#k-means performance:
			# print("ARI =", adjusted_rand_score(y, y_pred).round(2))
			# ARI = 0.76
			
			# #DBSCAN performance:
			# print("ARI =", adjusted_rand_score(y, clusters).round(2))
			# ARI = 0.99
		######################################################################################################################################################
		
		
		
		######################################################################################################################################################
		# Z-SCORE AND VELOCITY/GEOALTITUDE DISTRIBUTION JOINT PLOTS - SCATTER																				 #
		######################################################################################################################################################
		#Z-Score vs Velocity Distribution Joint Plot (Focused)
		if(z_score_joint_velocity_foc == True):

			#Create Plot
			g = sns.jointplot(data=points_master, x = "points_zscore", y = "velocity", legend = False, hue = "icao24", xlim=(-3.5, 3.5), ratio = 3)

			#Figure size
			g.fig.set_figwidth(10)
			g.fig.set_figheight(6.5)

			#Labels & Titles
			g.ax_joint.set_xlabel("Z-Score", weight = 'bold', fontsize = 12)
			g.ax_joint.set_ylabel("Velocity (meters/second)", weight = 'bold', fontsize = 12)
			plt.suptitle("Z-Score & Velocity Distribution (Focused) for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			fig = g.fig
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
		
		
		#Z-Score vs Altitude (Geo/GPS) Distribution Joint Plot (Focused)
		if(z_score_joint_geoaltitude_foc == True):

			#Create Plot
			g = sns.jointplot(data=points_master, x = "points_zscore", y = "geoaltitude", legend = False, hue = "icao24", xlim=(-3.5, 3.5), ratio = 3)

			#Figure size
			g.fig.set_figwidth(10)
			g.fig.set_figheight(6.5)

			#Labels & Titles
			g.ax_joint.set_xlabel("Z-Score", weight = 'bold', fontsize = 12)
			g.ax_joint.set_ylabel("Geo/GPS Altitude (meters)", weight = 'bold', fontsize = 12)
			plt.suptitle("Z-Score & Altitude (Geo/GPS) Distribution (Focused) for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			fig = g.fig
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
		######################################################################################################################################################



		######################################################################################################################################################
		#AWESOME JOINT PLOTS
		if(False == True):
			sns.jointplot(x = points_master['dropout_length'], y=points_master['velocity'], kind='scatter')
			sns.jointplot(x = points_master['dropout_length'], y=points_master['velocity'], kind='hex')
			sns.jointplot(x = points_master['dropout_length'], y=points_master['velocity'], kind='kde')
			sns.jointplot(x = points_master['dropout_length'], y=points_master['velocity'], kind='hist')
			plt.show()



		#2D Velocity-Dropout Countour Plot
		if(False == True):

			ax = sns.kdeplot(x = points_master['dropout_length'], y=points_master['velocity'])
			plt.show()

			# Custom the color, add shade and bandwidth
			sns.kdeplot(x = points_master['dropout_length'], y=points_master['velocity'], cmap="Greens", shade=True, bw_adjust=0.5)
			plt.show()

			# Add thresh parameter
			sns.kdeplot(x = points_master['dropout_length'], y=points_master['velocity'], cmap="Greens", shade=True, thresh=0)
			plt.show()
		######################################################################################################################################################

		
		
		######################################################################################################################################################
		# Z-SCORE VALUE PERCENTAGES																															 #
		######################################################################################################################################################
		#Z-Score Distribution Pie Chart - Improved
		if(z_score_distribution_pie == True):
			labels = [
				"0 to 0.9",
				"1 to 1.9",
				"2 to 2.9",
				"over 3"
			]

			pie_data = [
				points_master[(points_master['points_zscore'] > 0) & (points_master['points_zscore'] < 1)].reset_index(drop = True).copy().shape[0],		# > 0
				points_master[(points_master['points_zscore'] >= 1) & (points_master['points_zscore'] < 2)].reset_index(drop = True).copy().shape[0],		# >= 1
				points_master[(points_master['points_zscore'] >= 2) & (points_master['points_zscore'] < 3)].reset_index(drop = True).copy().shape[0],		# >= 2
				points_master[(points_master['points_zscore'] > 3)].reset_index(drop = True).copy().shape[0],												# > 3

			]


			
			fig, ax = plt.subplots(figsize=(10, 6.5), subplot_kw=dict(aspect="equal"))


			#Autopct
			def make_autopct(values):
				def my_autopct(pct):
					total = sum(values)
					val = int(round(pct*total/100.0))
					return '{p: .2f}%\n({v:d})'.format(p=pct,v=val)
				return my_autopct

			wedges, texts, = ax.pie(pie_data, explode = [0.05]*4, wedgeprops=dict(width=0.4), startangle=-40)#, autopct = make_autopct(pie_data), pctdistance=0.35)

			labels = [f'{l}, {s/sum(pie_data)*100:0.2f}%' for l, s in zip(labels, pie_data)]


			bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="w", lw=0.72)
			kw = dict(arrowprops=dict(arrowstyle="-"),
					bbox=bbox_props, zorder=0, va="center")
			
			for i, p in enumerate(wedges):
				ang = (p.theta2 - p.theta1)/2. + p.theta1
				y = np.sin(np.deg2rad(ang))
				x = np.cos(np.deg2rad(ang))
				horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
				connectionstyle = "angle,angleA=0,angleB={}".format(ang)
				kw["arrowprops"].update({"connectionstyle": connectionstyle})
				ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), horizontalalignment=horizontalalignment, **kw)

			plt.legend(wedges, labels, loc="lower left", bbox_to_anchor=(-0.25, 0))

			# # Change color of text
			# plt.rcParams['text.color'] = 'black'
			
			# # Create a circle at the center of the plot
			# my_circle=plt.Circle( (0,0), 0.65, color='white')
			
			# # Pieplot + circle on it
			# plt.pie(size, labels=names, autopct='%1.2f%%', pctdistance=0.5)#, explode = [0.05]*4, pctdistance = 0.5)#, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
			# p=plt.gcf()
			# p.gca().add_artist(my_circle)
			
			plt.text(x = -2, y = 1.25, s = "Skewness: " + str(stats.skew(points_master['dropout_length'].dropna())), weight = 'bold')

			plt.suptitle("Z-Score Value Distribution of Dropouts (Points > Average) for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tAverage: " + str("N/A") + "\tDropouts: " + str(num_dropouts_avg) + "\tPercent Dropouts: {:0.2f}%".format(pct_dropouts_avg)).expandtabs()
			#title += "\n[Ranges are inclusive of the outside value, ex. +2 would be is > 2 to 3"
			plt.title(title)
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()


		#Z-Score Distribution Pie Chart - OLD
		if(False == True):
			# Data
			#names = ["0 to 1", "1 to 2", "2 to 3", "above 3"]
			#size = [one_dev, two_dev, three_dev, over_three_dev]

			'''
			names = [
				"< -3",
				"<= -2",
				"<= -1",
				"< 0",
				"= 0",
				"> 0",
				">= +1",
				">= +2",
				"> +3"
			]

			size = [
				points_master[(points_master['points_zscore'] <= -3)].reset_index(drop = True).copy().shape[0],												# < -3
				points_master[(points_master['points_zscore'] > -3) & (points_master['points_zscore'] <= -2)].reset_index(drop = True).copy().shape[0],		# <= -2
				points_master[(points_master['points_zscore'] > -2) & (points_master['points_zscore'] <= -1)].reset_index(drop = True).copy().shape[0],		# <= -1
				points_master[(points_master['points_zscore'] > -1) & (points_master['points_zscore'] < -0)].reset_index(drop = True).copy().shape[0],		# < 0

				points_master[(points_master['points_zscore'] == 0)].reset_index(drop = True).copy().shape[0],												# = 0

				points_master[(points_master['points_zscore'] > 0) & (points_master['points_zscore'] < 1)].reset_index(drop = True).copy().shape[0],		# > 0
				points_master[(points_master['points_zscore'] >= 1) & (points_master['points_zscore'] < 2)].reset_index(drop = True).copy().shape[0],		# >= 1
				points_master[(points_master['points_zscore'] >= 2) & (points_master['points_zscore'] < 3)].reset_index(drop = True).copy().shape[0],		# >= 2
				points_master[(points_master['points_zscore'] > 3)].reset_index(drop = True).copy().shape[0],												# > 3

			]
			'''
			#return #JUST DO THE GROUPS OF THE PIE CHART WITH OPERATIONS INSTEAD OF CREATING VARIABLES. SHOW BOTH POSITIVE AND NEGATIVE DROPOUTS. THERE SHOULD BE 7 GROUPS IN THE PIE INCLUDING ZERO.
			names = [
				"0 to 0.9",
				"1 to 1.9",
				"2 to 2.9",
				"over 3"
			]

			size = [
				points_master[(points_master['points_zscore'] > 0) & (points_master['points_zscore'] < 1)].reset_index(drop = True).copy().shape[0],		# > 0
				points_master[(points_master['points_zscore'] >= 1) & (points_master['points_zscore'] < 2)].reset_index(drop = True).copy().shape[0],		# >= 1
				points_master[(points_master['points_zscore'] >= 2) & (points_master['points_zscore'] < 3)].reset_index(drop = True).copy().shape[0],		# >= 2
				points_master[(points_master['points_zscore'] > 3)].reset_index(drop = True).copy().shape[0],												# > 3

			]


			
			# create a figure and set different background
			fig = plt.figure()
			fig.patch.set_facecolor('white')
			
			# Change color of text
			plt.rcParams['text.color'] = 'black'
			
			# Create a circle at the center of the plot
			my_circle=plt.Circle( (0,0), 0.65, color='white')
			
			# Pieplot + circle on it
			plt.pie(size, labels=names, autopct='%1.2f%%', pctdistance=0.5)#, explode = [0.05]*4, pctdistance = 0.5)#, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
			p=plt.gcf()
			p.gca().add_artist(my_circle)
			
			#plt.text(x = -1.5, y = 1, s = "Skewness: " + str(stats.skew(points_master['dropout_length'].dropna())))

			plt.suptitle("Z-Score of Dropouts (Points > Average) for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tAverage: " + str("N/A") + "\tDropouts: " + str(num_dropouts_avg) + "\tPercent Dropouts: {:0.2f}%".format(pct_dropouts_avg)).expandtabs()
			#title += "\n[Ranges are inclusive of the outside value, ex. +2 would be is > 2 to 3"
			plt.title(title)
			plt.tight_layout()
			plt.show()
		######################################################################################################################################################


		
		######################################################################################################################################################
		#Z-SCORE DISTRIBUTION CHART
		######################################################################################################################################################
		if(z_score_dist_kde == True):
			ax = sns.displot(points_master, x = "points_zscore", kind = "kde", hue = "icao24", fill = True, legend = False)
			ax.set(xlim=(-3.5, 3.5))
			ax.set_xlabels("Points Z-Score")
			plt.suptitle("Points Z-Score Distribution (Focused) for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			plt.title(title)
			plt.tight_layout()
			
			#Figure size
			ax.fig.set_figwidth(10)
			ax.fig.set_figheight(6.5)

			#Display settings
			plt.tight_layout()
			#plt.show()
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
		######################################################################################################################################################
		
		
		
		######################################################################################################################################################
		#DROPOUT LENGTH DISTRIBUTION CHARTS
		######################################################################################################################################################
		#Dropout Length Distribution [Min - Max]
		if(dropout_length_dist_minmax == True):
			ax = sns.displot(points_master, x = "dropout_length", kind = "kde", hue = "icao24", fill = True)
			ax.set(xlim=(points_master['dropout_length'].min(), points_master['dropout_length'].max()))
			#ax.set(yscale="exp")
			#ax.set(xlim=(8, 12))
			ax._legend.remove()
			ax.set_xlabels("Dropout Length (seconds)")
			plt.suptitle("Dropout Length Distribution for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			title += "    Scale: [Min - Max]"
			title.expandtabs()
			plt.title(title)

			#Figure size
			ax.fig.set_figwidth(10)
			ax.fig.set_figheight(6.5)
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			fig = ax.ax.get_figure()
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
		
		
		#Dropout Length Distribution [8.5s to 11.5s]
		if(dropout_length_dist_focused == True):
			
			ax = sns.displot(points_master, x = "dropout_length", kind = "kde", hue = "icao24", fill = True)
			ax.set(xlim=(points_master['dropout_length'].min(), points_master['dropout_length'].max()))
			#ax.set(yscale="exp")
			ax.set(xlim=(8.5, 11.5))
			ax._legend.remove()
			ax.set_xlabels("Dropout Length (seconds)")
			plt.suptitle("Dropout Length Density for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			title += "    Scale: [8.5s to 11.5s]"
			title.expandtabs()
			plt.title(title)
			
			#Figure size
			ax.fig.set_figwidth(10)
			ax.fig.set_figheight(6.5)
			
			#Display settings
			plt.tight_layout()
			#plt.show()
			fig = ax.ax.get_figure()
			pdf.savefig(fig)
			plt.close("all")
			sns.reset_orig()
		######################################################################################################################################################
		
	#Holistic Metadata Average & Mode vs Number of Aircraft
	#ADDED TO PDF
	if(True == True):

		#Creat figure
		fig, ax = plt.subplots(figsize=(10, 6.5))

		#Plot data
		ax.plot('num_craft', 'avg', data = metadata, marker ='s', markerfacecolor = '#FF671F', label = 'Average', color = 'black')
		ax.plot('num_craft', 'mode', data = metadata, marker ='s', markerfacecolor = '#009A44', label = 'Mode', color = 'black')

		#Axis labels
		plt.xlabel('Number of Aircraft', weight = 'bold', fontsize = 12, color = '#009A44')
		plt.ylabel('Dropout Length (seconds)', weight = 'bold', fontsize = 12, color = '#009A44')
		plt.xticks(weight = 'bold', fontsize = 10)
		plt.yticks(weight = 'bold', fontsize = 10)
		
		#Titles
		plt.title("Holistic Point Info vs Number of Aircraft", weight = 'bold', color = '#009A44').set_fontsize('14')
		
		#Legend
		plt.legend()
		
		#Display settings
		plt.tight_layout()
		#plt.show()
		pdf.savefig(fig)

	
	points_master.to_csv("data/OpenSky/output/states_2022-01-17-10_output_master.csv")
	metadata.to_csv("data/OpenSky/output/states_2022-01-17-10_output_metadata.csv")
	return #JUST DO THE GROUPS OF THE PIE CHART WITH OPERATIONS INSTEAD OF CREATING VARIABLES. SHOW BOTH POSITIVE AND NEGATIVE DROPOUTS. THERE SHOULD BE 7 GROUPS IN THE PIE INCLUDING ZERO.



	




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
			
			
def append_files(directory, filename):
	
	
	pathslist = directory.glob('**/*.csv')
	
	data = pd.DataFrame()
	
	for path in pathslist:
		df = pd.read_csv(path)
		data = data.append(df)
	
	print(df.shape)
	data.to_csv(Path( "output/" + filename), index = False)
	return






#append_files(Path(Path.cwd() / "data/OpenSky/all_states 2022-01-17"), "states_2022-01-17-all.csv")
#exit(0)

#Print CWD
""" print("\nCWD: " + str(Path.cwd())) """

#Input file
infilename = "states_2022-01-17-10.csv"
infile = Path(Path.cwd() / "data/OpenSky/" / str(infilename))
""" print("\nFILE DIR: " + str(infile)) """
data = pd.read_csv(infile)

#Output file
outfilename = "states_2022-01-17-10_out_plots.pdf"
outfile = Path(Path.cwd() / "data/OpenSky/output/" / str(outfilename))
pdf = PdfPages(outfile)
""" print("\nFILE DIR: " + str(outfile)) """

#Number of aircraft to plot
#quantity = list([1, 10, 25, 50, 100])
quantity = list([1000])

#Run method
""" calc_dropouts(data, outfile, quantity) """
calc_dropouts(pd.read_csv(Path(Path.cwd() / "data/OpenSky/states_2022-01-17-all_agg_250.csv")), outfile, quantity)


#Close PDF
pdf.close()

#Aggregate data
#icao_matching(pd.read_csv(Path(Path.cwd() / "data/OpenSky/states_2022-01-17-all.csv")))


