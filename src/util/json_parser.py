"""
############################
University of North Dakota
JSON Parser
06/28/2021

Developers: Anton Skurdal


Description:
Simple program to read JSON
files and print them in the
console.
############################

"""


import json
import csv
import os
import numpy as np
import time
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist

#Make sure code runs as a module
if(__name__ == '__main__'):
	print("This code is meant to be run as a module.")
	exit(0)

def show_keys(console, path):
	
	console.delete(1.0, 'end')
	
	with open(path) as f:
		data = json.load(f)
	
	#console.insert(1.0, data.keys())
	#console.insert(1.0, "\n")
	
	s = ""
	s += str(data.keys()) + "\n"
	for key in data.keys():
		
		if(type(data[key]) != list):
			#console.insert(1.0, str(key) + ": " + str(data[key]))
			#console.insert(1.0, "\n")
			s += str(key) + ": " + str(data[key]) + "\n"
		else:
			s += str(key)
			for item in data[key]:
				#console.insert(1.0, item)
				#console.insert(1.0, "\n")
				
				
				if(type(item) == dict):
					s += json.dumps(item, indent=4, default=str)
				else:
					s += str(item) + "\n"
	
	console.insert(1.0, s)


def show_json(console, path):
	
	#console.delete(1.0, tk.END)
	
	#for child in console.winfo_children():
	#	child.destroy()
	
	with open(path) as f:
		data = json.load(f)	
	
	console.insert(1.0, json.dumps(data, indent = 4, sort_keys=False))
	
	"""
	print(data['aircraft'][0])
	
	
	for key in list(data.keys()):
		print(key)
		try:
			print(data[key].keys())
		except AttributeError:
			print("There are no nested keys in " + key + ".")
	
	
	ks = list(data.keys())
	print(ks)
	if (data[ks[0]].keys()):
		print("keys exist")
	print(data[ks[0]])
	x = "now"
	print(data[x])
	"""

def run_parse():
	
	"""
	fpath = "data\\original\\test2.json"
	
	floc = str(os.path.join(os.getcwd(), fpath))
	
	with open(floc) as f:
		data = json.load(f)
	
	print("Type: "+str(type(data))+"\n\n\n")
	
	print("Sample Output: "+str(data['meta']['view']['approvals'])+"\n\n\n")
	
	for key in sorted(data['meta']['view'].keys()):
		print(key)
	
	rev = data['meta']['view']['approvals'][0]
	#rev = rev.replace('reviewedAt', 'helloThere')
	print("REV: "+ str(type(rev)))
	print(str(rev.items()))
	print(str(rev['reviewedAt']))
	rev['reviewedAt'] = "hello"
	print(str(rev.get('reviewedAt')))
	print(str(rev.items()))
	
	
	
	print(json.dumps(data, indent = 4, sort_keys=True))
	"""
	
	
	folder = "data\\original\\readsb-hist2021-07-01"
	directory = str(os.path.join(os.getcwd(), folder))
	
	"""
	data = pd.read_json(os.path.join(os.getcwd(), "data\\test\\uasTest.json"))
	data.to_csv("uasTest.csv")
	print(data)
	
	
	exit()
	time.sleep(100)
	"""
	
	
	"""
	folder = "data\\test\\"
	
	directory = str(os.path.join(os.getcwd(), folder))
	
	
	with open(directory + "\\2021-01-01-000000Z.json", 'r') as j:
		contents = json.loads(j.read())
	
	
	print(contents['aircraft'])
	
	#json.loads(directory + "\\2021-01-01-000000Z.json")
	"""
	
	times = []
	a50c0e = []
	a2fcf2 = []
	rr_lat = []
	rr_lon = []
	
	lat = []
	lon = []
	
	target_hex = "a2fcf2"
	craft = []
	
	nav_altitude_mcp = []
	alt_baro = []
	
	
	#with open(directory+'_CSV\\'+target_hex+'.csv', 'w') as f:
		#w = csvwriter(f)
		#w.writerow(
		
		
	
	for file in os.listdir(directory):
		
		
		
		if(file.endswith(".json")):
			print(file)
			#with open(directory + "\\2021-01-01-000000Z.json", 'r') as j:
			with open(directory + "\\" + file, 'r') as j:
				contents = json.loads(j.read())
			times.append(contents['now'])
			
			for i in range(len(contents['aircraft'])):
				#print(i)
				#print(contents['aircraft'][i]['hex'])
				if(contents['aircraft'][i]['hex'] == target_hex):
					print(contents['aircraft'][i])
					craft.append(contents['aircraft'][i])
			
			
			#tm.sleep(100)
			##print(contents['aircraft'][0]['hex'])
			
			#a50ce.append(contents['aircraft'][0])
			#print(a50ce[0])
			#rr_lat.append(contents['aircraft'][0]['rr_lat'])
			#rr_lon.append(contents['aircraft'][0]['rr_lon'])		
			
			#a2fcf2.append(contents['aircraft'][1])
			#print(a2fcf2[0])
			#lat.append(contents['aircraft'][1]['lat'])
			#lon.append(contents['aircraft'][1]['lon'])		
			
			#print(str(data))
			#print(str(data.keys()))
			
	print(times)
	
	for i in range(len(times)):
		#times[i] = int(times[i])-1609459200
		times[i] = int(times[i])-1609459199
	
	
	#print(rr_lat)
	#print(rr_lon)
	
	#sns.lineplot(x=time, y=rr_lat)
	
	print(craft[0])
	
	
	folder = "data\\test\\"
	
	directory = str(os.path.join(os.getcwd(), folder))
	
	count = 0
	import itertools
	
	with open(directory + "\\" + target_hex + "_readsb-hist2021-07-01.csv", 'w', newline = '') as outfile:
		writer = csv.writer(outfile)
		for (i, j) in zip(craft, times):
			if(count == 0):
				header = list(i.keys())
				header.insert(0, 'time')
				writer.writerow(header)
				count += 1
			
			row = list(i.values())
			row.insert(0, j)
			#writer.writerow(i.values())
			writer.writerow(row)
	
	
	print("Finished writing to CSV.")
	
	#time.sleep(1000)
	
	for i in range(len(craft)):
		lat.append(craft[i]['lat'])
	
	for i in range(len(craft)):
		lon.append(craft[i]['lon'])
		
	for i in range(len(craft)):
		nav_altitude_mcp.append(craft[i]['nav_altitude_mcp'])
		
	for i in range(len(craft)):
		alt_baro.append(craft[i]['alt_baro'])
	
	
	
	print(lat)
	lat_lon = []
	
	for i in range(len(lat)):
		lat_lon.append(lat[i]/lon[i])
	
	
	
	df = {'Time':times, 'Lat':lat, 'Lon':lon, 'Lat/Lon':lat_lon, 'Altitude (MCP)': nav_altitude_mcp, 'Baro Altitude': alt_baro}
	df = pd.DataFrame(df)
	print(df)
	
	
	
	
	
	fig, host = plt.subplots(figsize = (8,5))
	
	par1 = host.twinx()
	par2 = host.twinx()
	par3 = host.twinx()
	par4 = host.twinx()
	#par5 = host.twinx()
	#par6 = host.twinx()
	
	#host.set_xlim(0, 2)
	#host.set_ylim(40, 60)
	#par1.set_ylim(0, 50)
	#par2.set_ylim(0, 50)
	
	
	host.set_xlabel("Time (Seconds Since 01-01-2021)")#, labelpad = 40)
	host.set_ylabel("Latitude")
	par1.set_ylabel("Longitude")
	par2.set_ylabel("Lat/Lon")
	par3.set_ylabel("Altitutude (MCP)")
	par4.set_ylabel("Barometric Altitude")
	
	
	colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
	color_iter = iter(colors)
	
	p1, = host.plot(df['Time'], df['Lat'], label="Latitude", color = next(color_iter))
	p2, = par1.plot(df['Time'], df['Lon'], label="Longitude", color = next(color_iter))
	p3, = par2.plot(df['Time'], df['Lat/Lon'], label="Lat/Lon", color = next(color_iter))
	p4, = par3.plot(df['Time'], df['Altitude (MCP)'], label="Altitutude (MCP)", color = next(color_iter))
	p5, = par3.plot(df['Time'], df['Baro Altitude'], label="Barometric Altitude", color = next(color_iter))
	
	#handles, labels = host.get_legend_handles_labels()
	#host.legend(handles, labels)
	#host.legend(loc='best')
	#host.legend(handles=lns, loc='best')
	##lns = [p1, p2, p3]
	#lns = [p1, p2]
	##lns=[p1]
	#host.legend(handles=lns, loc='best')
	
	# right, left, top, bottom
	par2.spines['right'].set_position(('outward', 60))
	par3.spines['right'].set_position(('outward', 120))
	par4.spines['right'].set_position(('outward', 180))
	
	color_iter = iter(colors)
	
	
	host.yaxis.label.set_color(color = next(color_iter))
	par1.yaxis.label.set_color(color = next(color_iter))
	par2.yaxis.label.set_color(color = next(color_iter))
	par3.yaxis.label.set_color(color = next(color_iter))
	par4.yaxis.label.set_color(color = next(color_iter))
	
	#host.yaxis.label.set_color(p1.get_color())
	#par1.yaxis.label.set_color(p2.get_color())
	#par2.yaxis.label.set_color(p3.get_color())
	#par3.yaxis.label.set_color(p4.get_color())
	#par4.yaxis.label.set_color(p5.get_color())
	
	plt.tight_layout()
	
	plt.title("Craft ICAO ID (6 hex digits): "+target_hex)
	plt.show()
	
	time.sleep(100)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	host = host_subplot(111, axes_class=axisartist.Axes)
	plt.subplots_adjust(right=0.75)
	
	
	
	par1 = host.twinx()
	par2 = host.twinx()
	par3 = host.twinx()
	
	par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(60, 0))
	par3.axis["right"] = par3.new_fixed_axis(loc="right", offset=(120, 0))
	
	par1.axis["right"].toggle(all=True)
	par2.axis["right"].toggle(all=True)
	par3.axis["right"].toggle(all=True)
	
	
	
	p1, = host.plot(df['Time'], df['Lat'], label="Latitude")
	p2, = par1.plot(df['Time'], df['Lon'], label="Longitude")
	p3, = par2.plot(df['Time'], df['Lat/Lon'], label="Lat/Lon")
	p4, = par3.plot(df['Time'], df['Altitude (MCP)'], label="Altitutude (MCP)")
	
	#host.set_xlim(0, 2)
	#host.set_ylim(0, 2)
	#par1.set_ylim(0, 4)
	#par2.set_ylim(1, 65)
	
	host.set_xlabel("Time (Seconds Since 01-01-2021)", labelpad = 40)
	host.set_ylabel("Latitude")
	par1.set_ylabel("Longitude")
	par2.set_ylabel("Lat/Lon")
	par3.set_ylabel("Altitutude (MCP)")
	
	plt.setp(host.axis["bottom"].major_ticklabels, rotation=-45)
	
	plt.title("Craft ICAO ID (6 hex digits): "+target_hex)
	
	#host.legend(loc='upper center')
	
	host.axis["bottom"].major_ticklabels.set_pad(8)
	host.axis["left"].label.set_color(p1.get_color())
	par1.axis["right"].label.set_color(p2.get_color())
	par2.axis["right"].label.set_color(p3.get_color())
	par3.axis["right"].label.set_color(p4.get_color())
	
	"""
	subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
	
	The parameter meanings (and suggested defaults) are:
	
	left  = 0.125  # the left side of the subplots of the figure
	right = 0.9    # the right side of the subplots of the figure
	bottom = 0.1   # the bottom of the subplots of the figure
	top = 0.9      # the top of the subplots of the figure
	wspace = 0.2   # the amount of width reserved for blank space between subplots
	hspace = 0.2   # the amount of height reserved for white space between subplots
	"""
	
	plt.tight_layout()
	#plt.subplots_adjust(left=0.125, bottom=0.15, right=0.6, top=0.9, wspace=0.4, hspace=0.4)
	plt.show()
	
	
	
	
	
	"""
	
	
	fig, ax = plt.subplots()
	ax2 = ax.twinx
	sns.lineplot(x=df['Time'], y=df['Lat'], ax=ax)
	#sns.lineplot(x=df['Time'], y=df['Lon'], ax=ax2)
	
	
	
	plt.show()
	"""
	
	
	#sns.lineplot(x=time, y=lat)
	#plt.show()
	
	#sns.lineplot(x=time, y=lat)
	#print(lat)
	#print(lon)
	
	#sns.lineplot(x=np.arange(len(time)), y=time)
	#plt.title("a50c0e")
	#plt.show()