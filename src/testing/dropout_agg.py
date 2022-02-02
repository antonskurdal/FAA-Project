import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
import numpy as np
import seaborn as sns
# sns.set_style('darkgrid', {"axes.facecolor": "lightgray"})
# sns.dark_palette("seagreen", as_cmap=True)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
np.set_printoptions(suppress=True)


#Set working path
import sys
base_dir = Path.resolve(Path.cwd())
sys.path.insert(0, base_dir)
print("\nBASE DIR: " + str(base_dir))



def calc_dropouts(data, outfile, quantity):

	if(isinstance(quantity, list) == False):
		quantity = str(quantity)
		quantity = map(int, quantity)
		quantity = list(quantity)
	
	print(quantity)
	for q in quantity:
		print("Q: " + str(q))

		#Format lastcontact to not be in scientific notation
		data['lastcontact'] = data['lastcontact'].astype(int)

		#Create a master dataframe
		points_master = pd.DataFrame()

		craft_list = data['icao24'].unique()
		print("LENGTH OF CRAFT LIST: " + str(len(craft_list)))
		craft_list = craft_list[:q]


		for craft in craft_list:
			

			#craft = data['icao24'].unique()[0]
			#print("\nCRAFT: " + str(craft))

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
		




		####################################################################################################
		#Reset master index - NOTE: ALL NAN POINTS, WHICH ARE THE FIRST POINTS OF EACH CRAFT DATA SUBSET, ARE DROPPED HERE
		#print(points_master.shape)
		points_master = points_master.dropna()
		#print(points_master.shape)
		points_master.reset_index(inplace = True)
		#print(points_master.head(5))


		'''
		#Reset master dataframe index
		points_master.reset_index(inplace=True)
		'''
		#print(points_master)
		#print(points_master['points_mode'].unique())

		#Number of unique aircraft
		num_craft = points_master['icao24'].unique().shape[0]

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

		#Print calculated data
		print("Points: " + str(num_points))
		print(">Points Average: " + str(points_master['points_avg'][0]))
		print(">Points Mode: " + str(points_master['dropout_length'].mode()[0]))
		print("Dropouts [Points > Average]: " + str(num_dropouts_avg))
		print(">Percentage: {:.4f}%".format(pct_dropouts_avg))

		print("Dropouts [Points > Mode]: " + str(num_dropouts_mode))
		print(">Percentage: {:.4f}%".format(pct_dropouts_mode))


		#one_dev = points_master[(points_master['dropout_length'] > points_master['dropout_avg'][0]) & (points_master['dropout_zscore'] > 0) & (points_master['dropout_zscore'] <= 1)].copy().shape[0]
		#print(points_master)
		#print(num_points)
		
		#abv_avg = points_master[(points_master['dropout_length'] > points_master['points_avg'][0])].reset_index(drop = True).copy()
		abv_avg = points_master.copy()
		#print(abv_avg.head(500))
		below_zero_dev = abv_avg[(abv_avg['points_zscore'] <= 0)].reset_index(drop = True).copy().shape[0]
		#one_dev = points_master[(points_master['dropout_length'] > points_master['points_avg'][0])].reset_index(drop = True).copy()
		one_dev = abv_avg[(abv_avg['points_zscore'] > 0) & (abv_avg['points_zscore'] <= 1)].reset_index(drop = True).copy().shape[0]

		two_dev = abv_avg[(abv_avg['points_zscore'] > 1) & (abv_avg['points_zscore'] <= 2)].reset_index(drop = True).copy().shape[0]

		three_dev = abv_avg[(abv_avg['points_zscore'] > 2) & (abv_avg['points_zscore'] <= 3)].reset_index(drop = True).copy().shape[0]
		over_three_dev = abv_avg[(abv_avg['points_zscore'] > 3)].reset_index(drop = True).copy().shape[0]

		print("\nStandard Deviation Counts:")
		print("[TOTAL]: " + str(one_dev + two_dev + three_dev + over_three_dev))
		print("[<=  0]: " + str(below_zero_dev))
		print("[<= +1]: " + str(one_dev))
		print("[<= +2]: " + str(two_dev))
		print("[<= +3]: " + str(three_dev))
		print("[>  +3]: " + str(over_three_dev))


		#Holistic Z-Score
		#zscores = stats.zscore(list(points_master['dropout_length'].dropna()))
		points_master['holistic_zscore'] = (points_master.dropout_length - points_master.dropout_length.mean())/points_master.dropout_length.std(ddof=0)
		print(points_master['holistic_zscore'].head(5))
		print(points_master.columns)
		print(points_master['points_zscore'].describe().apply(lambda x: format(x, 'f')))
		
		#zscores = np.insert(zscores, 0, np.NaN, axis = 0)
		#points_master.insert(points_master.shape[1], 'holistic_zscore', zscores)





		#Holistic Percentage Dropouts Pie Charts - Average and Mode
		if(True == True):


			fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 6.5))
			
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

			import matplotlib.patheffects as path_effects
			ax1.text(0, 0, "Average:\n{:.2f}".format(points_master['dropout_length'].mean()), ha='center', va='center', fontsize=28, weight = 'normal', color = 'black')#.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
			ax2.text(0, 0, "Mode:\n{:.2f}".format(points_master['dropout_length'].mode()[0]), ha='center', va='center', fontsize=28, weight = 'normal', color = 'black')#.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
			#ax1.legend(loc = 'lower center')
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=False, ncol=2, handles = [drop_patch, norm_patch])
			ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=False, ncol=2, handles = [drop_patch, norm_patch])
			
			#ax1.set_frame_on(True)
			#ax1.set_facecolor("#AEAEAE")
			#plt.setp(ax1.spines.values(),visible=False)
			
			ax1.set_title("Points > Average", weight = 'bold', fontsize = 14)
			ax2.set_title("Points > Mode", weight = 'bold', fontsize = 14)

			fig.set_facecolor("#AEAEAE")

			plt.suptitle("Holistic Percentage of Dropouts for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tAverage: " + str("N/A") + "\tDropouts: " + str(num_dropouts_avg) + "\tPercent Dropouts: {:0.2f}%".format(pct_dropouts_avg)).expandtabs()
			#title += "\n[Ranges are inclusive of the outside value, ex. +2 would be is > 2 to 3"
			#plt.title(title)
			plt.tight_layout()
			plt.show()



		#Percentage Dropouts Pie Charts - Average and Mode
		if(True == True):


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



		#Holistic Z-Score Distribution Plot - Kernel Density Estimates
		if(False == True):

			ax = sns.displot(points_master, x = "holistic_zscore", kind = "kde", hue = "icao24", fill = True, legend = False)
			
			
			#ax = sns.displot(points_master, x = "holistic_zscore", kde = True)#, hue = "icao24", fill = True, legend = False)
			#ax.set(xlim=(-3.5, 3.5))
			ax.set(xlim=(points_master['holistic_zscore'].min()-1, points_master['holistic_zscore'].max()+1))
			#ax.set(yscale="log")
			#ax.set(ylim=(0, 100))
			#print(plt.ylim())

			desc = points_master['holistic_zscore'].describe()
			
			desc = desc.apply(lambda x: format(x, '.4f'))
			#desc['count'] = int(desc['count'].astype(int)
			desc['count'] = int(pd.to_numeric(desc['count']))
			print(desc)
			data1=[i for i in desc.index]
			data2=[str(i) for i in desc]
			text= ('\n'.join([ a +':'+ b for a,b in zip(data1,data2)]))
			#plt.text(40, 0.1, text , fontsize=12)
			#from matplotlib import rc

			# activate latex text rendering
			#rc('text', usetex=True)
			leg = plt.legend(labels = [text], title = "Point Info", loc = 'best', bbox_to_anchor = (1,1), title_fontproperties={'weight':'bold', 'size':10}, markerscale = 0)
			#leg = plt.legend(handlelength=0, handletextpad=0, fancybox=True)
			for item in leg.legendHandles:
				item.set_visible(False)
			
			for t in leg.get_texts():
				t.set_ha('left')
				#leg._legend_box.align = "left"
			

			ax.fig.set_figwidth(12)
			ax.fig.set_figheight(8)
			ax.set_xlabels("Z-Score")
			plt.suptitle("Holistic Z-Score Kernel Density Estimates Distribution", weight = 'bold').set_fontsize('16')
			title = ("Number of Aircraft: " + str(num_craft) + "\tData Points: " + str(num_points)).expandtabs()# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			plt.title(title)
			plt.tight_layout()
			plt.show()
			#pdf.savefig()
			#plt.close("all")


		#Holistic Z-Score Distribution Plot - Empirical Cumulative
		if(False == True):
				
				
				ax = sns.displot(points_master, x = "holistic_zscore", kind = "ecdf")#, hue = "icao24", fill = True, legend = False)
				'''
				ax = sns.displot(points_master, x = "holistic_zscore", kind = "kde", hue = "icao24", fill = True, legend = False)
				'''
				
				#ax = sns.displot(points_master, x = "holistic_zscore", kde = True)#, hue = "icao24", fill = True, legend = False)
				#ax.set(xlim=(-3.5, 3.5))
				ax.set(xlim=(points_master['holistic_zscore'].min()-1, points_master['holistic_zscore'].max()+1))
				#ax.set(yscale="log")
				#ax.set(ylim=(0, 100))
				#print(plt.ylim())

				desc = points_master['holistic_zscore'].describe()
				
				desc = desc.apply(lambda x: format(x, '.4f'))
				#desc['count'] = int(desc['count'].astype(int)
				desc['count'] = int(pd.to_numeric(desc['count']))
				print(desc)
				data1=[i for i in desc.index]
				data2=[str(i) for i in desc]
				text= ('\n'.join([ a +':'+ b for a,b in zip(data1,data2)]))
				#plt.text(40, 0.1, text , fontsize=12)
				#from matplotlib import rc

				# activate latex text rendering
				#rc('text', usetex=True)
				leg = plt.legend(labels = [text], title = "Point Info", loc = 'best', bbox_to_anchor = (1,1), title_fontproperties={'weight':'bold', 'size':10}, markerscale = 0)
				#leg = plt.legend(handlelength=0, handletextpad=0, fancybox=True)
				for item in leg.legendHandles:
					item.set_visible(False)
				
				for t in leg.get_texts():
					t.set_ha('left')
					#leg._legend_box.align = "left"
				

				ax.fig.set_figwidth(12)
				ax.fig.set_figheight(8)
				ax.set_xlabels("Z-Score")
				plt.suptitle("Holistic Z-Score Empirical Cumulative Distribution", weight = 'bold').set_fontsize('16')
				title = ("Number of Aircraft: " + str(num_craft) + "\tData Points: " + str(num_points)).expandtabs()# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
				plt.title(title)
				plt.tight_layout()
				plt.show()
				#pdf.savefig()
				#plt.close("all")






		#Holistic Z-Score Distribution Plot - Histogram
		if(False == True):
			
			
			ax = sns.displot(points_master, x = "holistic_zscore", kind = "hist", bins = 50, log_scale=(False, True))#, hue = "icao24", fill = True, legend = False)
			'''
			ax = sns.displot(points_master, x = "holistic_zscore", kind = "ecdf")#, hue = "icao24", fill = True, legend = False)
			ax = sns.displot(points_master, x = "holistic_zscore", kind = "kde", hue = "icao24", fill = True, legend = False)
			
			'''
			
			#ax = sns.displot(points_master, x = "holistic_zscore", kde = True)#, hue = "icao24", fill = True, legend = False)
			#ax.set(xlim=(-3.5, 3.5))
			ax.set(xlim=(points_master['holistic_zscore'].min()-1, points_master['holistic_zscore'].max()+1))
			#ax.set(yscale="log")
			#ax.set(ylim=(0, 100))
			#print(plt.ylim())

			desc = points_master['holistic_zscore'].describe()
			
			desc = desc.apply(lambda x: format(x, '.4f'))
			#desc['count'] = int(desc['count'].astype(int)
			desc['count'] = int(pd.to_numeric(desc['count']))
			print(desc)
			data1=[i for i in desc.index]
			data2=[str(i) for i in desc]
			text= ('\n'.join([ a +':'+ b for a,b in zip(data1,data2)]))
			#plt.text(40, 0.1, text , fontsize=12)
			#from matplotlib import rc

			# activate latex text rendering
			#rc('text', usetex=True)
			leg = plt.legend(labels = [text], title = "Point Info", loc = 'best', bbox_to_anchor = (1,1), title_fontproperties={'weight':'bold', 'size':10}, markerscale = 0)
			#leg = plt.legend(handlelength=0, handletextpad=0, fancybox=True)
			for item in leg.legendHandles:
				item.set_visible(False)
			
			for t in leg.get_texts():
				t.set_ha('left')
				#leg._legend_box.align = "left"
			

			ax.fig.set_figwidth(12)
			ax.fig.set_figheight(8)
			ax.set_xlabels("Z-Score")
			plt.suptitle("Holistic Z-Score Distribution", weight = 'bold').set_fontsize('16')
			title = ("Number of Aircraft: " + str(num_craft) + "\tData Points: " + str(num_points)).expandtabs()# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			plt.title(title)
			plt.tight_layout()
			plt.show()
			#pdf.savefig()
			#plt.close("all")


		#Sklearn Clustering Plots
		if(False == True):
				

			from sklearn.cluster import DBSCAN
			from sklearn import metrics
			from sklearn.datasets import make_blobs
			from sklearn.preprocessing import StandardScaler

			#X = np.array(points_master.loc(axis=0)[:, :, 'points_zscore', 'velocity'].dropna())
			X = points_master[['points_zscore', 'velocity']].dropna().to_numpy()

			# cluster the data into five clusters
			from sklearn.cluster import KMeans
			kmeans = KMeans(n_clusters=5)
			kmeans.fit(X)
			y_pred = kmeans.predict(X)# plot the cluster assignments and cluster centers
			plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="plasma")
			plt.scatter(kmeans.cluster_centers_[:, 0],   
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
			plt.tight_layout()
			plt.show()










			scaler = StandardScaler()
			X_scaled = scaler.fit_transform(X)

			dbscan = DBSCAN(eps = 0.123, min_samples = 2)
			clusters = dbscan.fit_predict(X_scaled)
			# plot the cluster assignments
			plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="plasma")
			plt.xlabel("Point Z-Score")
			plt.ylabel("Velocity (m/s)")
			plt.suptitle("DBSCAN Clustering: Z-Score & Velocity", weight = 'bold').set_fontsize('16')
			plt.title("Number of Aircraft: " + str(num_craft))
			plt.tight_layout()
			plt.show()
			#clustering.labels_

			# from sklearn.metrics.cluster import adjusted_rand_score#k-means performance:
			# print("ARI =", adjusted_rand_score(y, y_pred).round(2))
			# ARI = 0.76
			
			# #DBSCAN performance:
			# print("ARI =", adjusted_rand_score(y, clusters).round(2))
			# ARI = 0.99








		#Join plot with distribution at a specific range
		if(False == True):
			sns.jointplot(data=points_master, x = "points_zscore", y = "velocity", legend = False, hue = "icao24", xlim=(-3.5, 3.5), ratio = 3)
			plt.show()



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


		#Z-Score Distribution Pie Chart - Improved
		if(False == True):
			labels = [
				"0 to 0.9",
				"1 to 1.9",
				"2 to 2.9",
				"over 3"
			]

			data = [
				points_master[(points_master['points_zscore'] > 0) & (points_master['points_zscore'] < 1)].reset_index(drop = True).copy().shape[0],		# > 0
				points_master[(points_master['points_zscore'] >= 1) & (points_master['points_zscore'] < 2)].reset_index(drop = True).copy().shape[0],		# >= 1
				points_master[(points_master['points_zscore'] >= 2) & (points_master['points_zscore'] < 3)].reset_index(drop = True).copy().shape[0],		# >= 2
				points_master[(points_master['points_zscore'] > 3)].reset_index(drop = True).copy().shape[0],												# > 3

			]


			
			fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

			wedges, texts = ax.pie(data, explode = [0.05]*4, wedgeprops=dict(width=0.4), startangle=-40)

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

				

			# # Change color of text
			# plt.rcParams['text.color'] = 'black'
			
			# # Create a circle at the center of the plot
			# my_circle=plt.Circle( (0,0), 0.65, color='white')
			
			# # Pieplot + circle on it
			# plt.pie(size, labels=names, autopct='%1.2f%%', pctdistance=0.5)#, explode = [0.05]*4, pctdistance = 0.5)#, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
			# p=plt.gcf()
			# p.gca().add_artist(my_circle)
			
			# #plt.text(x = -1.5, y = 1, s = "Skewness: " + str(stats.skew(points_master['dropout_length'].dropna())))

			plt.suptitle("Z-Score of Dropouts (Points > Average) for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tAverage: " + str("N/A") + "\tDropouts: " + str(num_dropouts_avg) + "\tPercent Dropouts: {:0.2f}%".format(pct_dropouts_avg)).expandtabs()
			#title += "\n[Ranges are inclusive of the outside value, ex. +2 would be is > 2 to 3"
			plt.title(title)
			plt.tight_layout()
			plt.show()


		#Z-Score Distribution Pie Chart
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



		

		#Z-Score Distribution Chart
		if(False == True):
			ax = sns.displot(points_master, x = "points_zscore", kind = "kde", hue = "icao24", fill = True, legend = False)
			ax.set(xlim=(-3.5, 3.5))
			ax.set_xlabels("Points Z-Score")
			plt.suptitle("Points Z-Score Distribution for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			plt.title(title)
			plt.tight_layout()
			plt.show()
			#pdf.savefig()
			#plt.close("all")

		#Dropout Length Distribution [Mix - Max]
		if(False == True):
			ax = sns.displot(points_master, x = "dropout_length", kind = "kde", hue = "icao24", fill = True)
			ax.set(xlim=(points_master['dropout_length'].min(), points_master['dropout_length'].max()))
			#ax.set(yscale="exp")
			#ax.set(xlim=(8, 12))
			ax._legend.remove()
			ax.set_xlabels("Dropout Length (seconds)")
			plt.suptitle("Dropout Length Density for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
			title = ("Points: " + str(num_points))# + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
			title += "    Scale: [Min - Max]"
			title.expandtabs()
			plt.title(title)
			plt.tight_layout()
			plt.show()
			#pdf.savefig()
			#plt.close("all")

		if(False == True):
			#Dropout Length Distribution [8.5s to 11.5s]
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
			plt.tight_layout()
			plt.show()
			#pdf.savefig()
			#plt.close("all")

		


		
		'''
		#one_dev = len(points_master[(points_master['dropout_zscore_round'] >= -1) & (points_master['dropout_zscore_round'] <= 1)])
		two_dev = len(points_master[(points_master['dropout_zscore_round'] >= -2) & (points_master['dropout_zscore_round'] <= 2)]) - one_dev
		three_dev = len(points_master[(points_master['dropout_zscore_round'] >= -3) & (points_master['dropout_zscore_round'] <= 3)]) - one_dev - two_dev
		rest_dev = len(points_master['dropout_zscore_round']) - one_dev - two_dev - three_dev
		print(one_dev)
		print(two_dev)
		print(three_dev)
		print(rest_dev)


		# Data
		names = 'One', 'Two', 'Three', '> Three',
		size = [one_dev, two_dev, three_dev, rest_dev]
		
		# create a figure and set different background
		fig = plt.figure()
		fig.patch.set_facecolor('white')
		
		# Change color of text
		plt.rcParams['text.color'] = 'black'
		
		# Create a circle at the center of the plot
		my_circle=plt.Circle( (0,0), 0.6, color='white')
		
		# Pieplot + circle on it
		plt.pie(size, labels=names, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
		p=plt.gcf()
		p.gca().add_artist(my_circle)


		plt.suptitle("Dropout Standard Deviations from Mean for " + str(num_craft) + " Aircraft", weight = 'bold').set_fontsize('16')
		title = ("Points: " + str(num_points) + "\tDropouts: " + str(num_points) + "\tPercent Dropouts: {:0.2f}%".format(pct_points)).expandtabs()
		plt.title(title)
		#plt.tight_layout()
		plt.show()



		#Percentages
		'''
	return #JUST DO THE GROUPS OF THE PIE CHART WITH OPERATIONS INSTEAD OF CREATING VARIABLES. SHOW BOTH POSITIVE AND NEGATIVE DROPOUTS. THERE SHOULD BE 7 GROUPS IN THE PIE INCLUDING ZERO.



	




print("\nCWD: " + str(Path.cwd()))

#Input file
infilename = "states_2022-01-17-10.csv"
infile = Path(Path.cwd() / "data/OpenSky/" / str(infilename))
print("\nFILE DIR: " + str(infile))
data = pd.read_csv(infile)

#Output file
outfilename = "states_2022-01-17-10_droputs.pdf"
outfile = Path(Path.cwd() / "data/OpenSky/output/" / str(outfilename))
print("\nFILE DIR: " + str(outfile))

#Number of aircraft to plot
quantity = 1


pdf = PdfPages(outfile)
calc_dropouts(data, outfile, [10])
#calc_dropouts(data, pdf, quantity)
#calc_dropouts(data, pdf, 5)
#calc_dropouts(data, outfile, 10)
#calc_dropouts(data, outfile, 25)
# calc_dropouts(data, outfile, 50)
#calc_dropouts(data, outfile, 100)
# calc_dropouts(data, outfile, 7304)

pdf.close()