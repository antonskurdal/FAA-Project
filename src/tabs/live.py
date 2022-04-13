#!/usr/bin/env python

"""This file controls the Live Tab.

	Live Tab description.
"""

import tkinter as tk
from tkinter import *
from tkinter import messagebox
from pathlib import Path
import pandas as pd
import _thread
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import cartopy.crs as crs
import cartopy.feature as cfeature
from isort import api
import ast

from opensky_api import OpenSkyApi

import util.sku_widgets as sku

__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2020, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"

# Make sure code runs as a module
if(__name__ == '__main__'):
	print("This code is meant to be run as a module.")
	exit(0)


#Padding
PADX_CONFIG = (2, 2)
PADY_CONFIG = (2, 2)

class LiveData(tk.Frame):

	def __init__(self, parent, controller, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs, bg = sku.FRAME_BACKGROUND)
		self.controller = controller
		
		#Grid Management
		for row in range(7):
			self.grid_rowconfigure(row, weight = 1)
		for col in range(18):
			self.grid_columnconfigure(col, weight = 1)
		
		for row in range(7):
			self.grid_rowconfigure(row, weight = 0, minsize = 100)
		for col in range(18):
			self.grid_columnconfigure(row, weight = 0, minsize = 100)

		###########
		# METHODS #
		###########
		def switch(master):
		
			# Determine if switch is on or off
			if self.is_on:
				
				labelframe_live_switch = sku.CustomLabelFrame(master, text = "Rolling", labelanchor = 'n')
				labelframe_live_switch.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
				labelframe_live_switch.grid_anchor('center')
				
				switch_live = sku.BorderButton(labelframe_live_switch, button_image = self.off, button_command = lambda: [switch(master)], button_activebackground = '#404040')
				switch_live.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
				
				# label_live.config(label_live.child_text.set("The Switch is Off"), bg = "red")
				self.is_on = False
				labelframe_live_switch['text'] = "Off"
				#labelframe_live_switch['text'] = str(self.is_on)
				self.button_track.child['state'] = 'normal'
			else:
				
				labelframe_live_switch = sku.CustomLabelFrame(master, text = "Rolling", labelanchor = 'n')
				labelframe_live_switch.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
				labelframe_live_switch.grid_anchor('center')
				
				switch_live = sku.BorderButton(labelframe_live_switch, button_image = self.on, button_command = lambda: [switch(master)], button_activebackground = '#404040')
				switch_live.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
				
				# label_live.config(label_live.child_text.set("The Switch is On"), bg = "green")
				self.is_on = True
				labelframe_live_switch['text'] = "On"
				#labelframe_live_switch['text'] = str(self.is_on)
				self.button_track.child['state'] = 'disabled'
				


		

		global craft_icao24
		self.count = 0
		global lat
		global lon
		


		def append_df(df, craft):
			if (df.empty):
				craft_dict = ast.literal_eval(str(craft))
				print(type(craft_dict))
				print(craft_dict)



		def get_states(console, api):
			
			states = api.get_states()
			#print(states)

			'''
			for s in states.states:
				print(s.icao24)
			'''

			craft = states.states[0]

			for s in states.states:
				print(s)
				if(s.origin_country == "United States"):
					#print(s)
					craft = s
					break

			global craft_icao24
			craft_icao24 = craft.icao24

			"""
			print(craft.time_position)
			print("CHOSEN CRAFT: " + str(craft))
			print("TIME POS: " + str(craft.time_position))
			"""

			console.insert(tk.END, "Chosen Aircraft: " + str(craft) + "\n")

			global count
			count = 0

			global lat
			global lon
			lat = []
			lon = []

			self.button_track.child['state'] = 'normal'
		
		def live_data():
			
			threading.Timer(10, live_data).start()
			global count
			data = api.get_states(icao24 = craft_icao24)
			print("\nRetrieval #" + str(count))
			count = count + 1
			print(type(data))
			if (data is None):
				print("EMPTY PACKET")
				return
			elif (isinstance (data, type("NoneType")) == False):
				#print("TIME POS: " + str(data))
				print(data.states[0].longitude)
				lon.append(data.states[0].longitude)
				lat.append(data.states[0].latitude)
				#updateplot(lat, lon)
				

			else:
				print("ERROR")
		
		def plotting_thread(fig_a, ax_a, fig_b, ax_b, console):
			while (self.is_on == True):
				self.button_track.child['state'] = 'disabled'

				time.sleep(1)  # ... or some busy computing
				ax_a.clear()
				ax_b.clear()

				data = self.api.get_states(icao24 = craft_icao24)
				
				
				#print("\nRetrieval #" + str(count))
				count_string = "{:0>3d}".format(self.count)
				console.insert(tk.END, "\n\n-----------------------")
				console.insert(tk.END, "\n| Retrieval #" + count_string + " |")
				console.insert(tk.END, "\n-----------------------")
				
				self.count = self.count + 1
				#print(type(data))
				console.insert(tk.END, "\nCLASS: " + str(type(data)))
				
				if (data is None):
					#print("EMPTY PACKET")
					#console.insert(tk.END, "\nCLASS: empty packet")
					console.see(tk.END)
					continue
				#return




				elif (isinstance (data, type("NoneType")) == False):
					#print("TIME POS: " + str(data))
					#print(data.states[0].longitude)

					#append_df(self.df, data.states[0])
					if(self.df.empty):
						craft = data.states[0]
						craft_dict = ast.literal_eval(str(craft))
						self.df = self.df.from_records(craft_dict, index=[0])
						#print(self.df)
					else:
						craft = data.states[0]
						craft_dict = ast.literal_eval(str(craft))
						temp = pd.DataFrame.from_records(craft_dict, index=[0])
						self.df = self.df.append(temp, ignore_index=True)
						#print(self.df)
					
					x = "last_contact"
					console.insert(tk.END, "\nLAST_CONTACT: " + str(data.states[0].__getattribute__(x)))
					lon.append(data.states[0].longitude)
					lat.append(data.states[0].latitude)
					#updateplot(lat, lon)


					#print(data.states[0])
				else:
					#print("ERROR")
					console.insert(tk.END, "ERROR")
				
				#Adjust console view
				console.see(tk.END)
				
				
				#Dropout Plot
				dropouts = pd.DataFrame()
				dropouts.insert(0, 'last_contact', self.df['last_contact'])
				dropouts.insert(0, 'dropout_length', self.df['last_contact'].diff()[1:])
				#print(dropouts)
				dropout_mean = dropouts['dropout_length'].mean()
				
				if (dropouts['dropout_length'].shape[0] == 1):
					dropout_mode = dropout_mean
				else:
					dropout_mode = dropouts['dropout_length'].mode()[0]
				# dropout_mode = dropouts['dropout_length'].mode()
				print("[LIVE] Dropout Mean: {}".format(dropout_mean))
				print("[LIVE] Dropout Mode: {}".format(dropout_mode))
				#ax_b.plot(self.df.index, dropouts['dropout_length'])
				
				from matplotlib import cm
				ax_b.plot(self.df.index, dropouts['dropout_length'], linestyle = '-', color = 'gray', zorder = 1)
				ax_b.scatter(self.df.index, dropouts['dropout_length'], c=cm.RdYlGn_r(dropouts['dropout_length']/dropouts['dropout_length'].max()), zorder = 2, edgecolors = 'gray')
				ax_b.set_ylim(ax_b.get_ylim()[::-1])
				
				# x = self.df.index.to_numpy(copy = True)
				# y = dropouts['dropout_length'].to_numpy(copy = True)
				# upper = np.ma.masked_where(y < dropout_mean, y)
				# lower = np.ma.masked_where(y > dropout_mean, y)
				# middle = np.ma.masked_where((y < lower) | (y > upper), y)


				# ax_b.plot(x, middle, x, lower, x, upper, x)
				'''
				import matplotlib as mpl
				cmap = mpl.cm.RdYlGn_r
				norm = mpl.colors.Normalize(vmin=5, vmax=10)
				fig_b.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_b, orientation='horizontal', label='Some Units')
				'''
				'''
				cax = fig_b.add_axes([0, 0, 0, 0])
				im = ax_b.imshow([self.df.index, dropouts['dropout_length']], cmap='RdYlGn_r')
				fig_b.colorbar(im, cax=cax, orientation='vertical')
				'''

				import matplotlib as mpl
				norm = mpl.colors.Normalize(vmin=0, vmax=1000000)
				cmap = plt.cm.RdYlGn
				cax = fig_b.add_axes([0.92, 0.12, 0.02, 0.75])
				cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, spacing='proportional')
				cb.set_label('Dropout Severity')
				cb.set_ticks([])



				mean_line = ax_b.axhline(dropout_mean, xmin = 0, xmax = 1, label='Mean', linestyle='--', color = "pink")
				mode_line = ax_b.axhline(dropout_mode, xmin = 0, xmax = 1, label='Mode', linestyle='--', color = "purple")
				#ax_b.axhline(dropout_mean, xmin = 0, xmax = 1, label='Mean', linestyle='--', color = "#FF671F")
				#ax_b.axhline(dropout_mode, xmin = 0, xmax = 1, label='Mode', linestyle='--', color = "#009A44")
				ax_b.legend(handles = [mean_line, mode_line], labels = ["Mean", "Mode"], loc = "upper right", bbox_to_anchor=(0.525, 0.675, 0.6, 0.5), ncol = 2)
				
				fig_b.canvas.draw_idle()  # use draw_idle instead of draw
				
				ax_b.set_xlabel("Packet Number")
				ax_b.set_ylabel("Time Since Last Contact (seconds)")
				ax_b.set_title("Country: " + data.states[0].origin_country +" icao24: " + str(craft_icao24) + " Live Dropouts")
				
				


				
				
				
				
				
				
				
				
				#plt.style.use(['dark_background'])
				
				'''
				#Normal Plot
				ax.scatter(lon, lat)
				'''

				#CartoPy Plot
				ax_a.scatter(lon, lat, transform = crs.PlateCarree())
				ax_a.set_global()
				ax_a.stock_img()
				ax_a.coastlines()
				ax_a.add_feature(cfeature.COASTLINE)
				ax_a.add_feature(cfeature.STATES)# Zoom in on the US by setting longitude/latitude parameters
				
				
				
				
				
				ax_a.set_extent(
					[
						-135, # minimum latitude
						-60, # min longitude
						20, # max latitude
						55 # max longitude
					],
					crs=crs.PlateCarree()
				)
				# ax.set_extent(
				# 	[
				# 		lat[0], # minimum latitude
				# 		lon[0], # min longitude
				# 		lat[-1]+1, # max latitude
				# 		lon[-1]+1 # max longitude
				# 	],
				# 	crs=crs.PlateCarree()
				# )

				ax_a.set_xlabel("Latitude")
				ax_a.set_ylabel("Longitude")
				ax_a.set_title("Country: " + data.states[0].origin_country +" icao24: " + str(craft_icao24) + " Live Position")




				fig_a.canvas.draw_idle()  # use draw_idle instead of draw
		
		
		
		
		# OpenSky Network Login Management
		self.username = tk.StringVar()
		self.password = tk.StringVar()
		try:
			import util.login.login as login
			self.username.set(login.username)
			self.password.set(login.password)
			
		except ImportError:
			#raise ImportError("[LIVE] Login not found.")
			messagebox.showerror(title="Error", message=
				"[LIVE] Login not found."+
				
				"\n\nPlease add your login information by following these steps:"+
				"\n 1. Navigate to src/util/login"+
				"\n 2. Create a file called 'login.py' inside the login folder"+
				"\n 3. Define string variables with login information as follows:"+
				"\n\t a)username = <username>"+
				"\n\t b)password = <password>"+
				"\n 4. Relaunch the program."+
				"\n\nYour username and password will then be automatically used as the default OpenSky Network login."+
				
				"\n\nAlternatively, navigate to src/tabs/live.py, find this exception code, and change the defaults below this line.")
			
			
			""" print("\n[LIVE] Login not found.")
			print("Please add your login information by following these steps:")
			print("\t1. Navigate to src/util/login")
			print("\t2. Create a file called 'login.py' inside the login folder")
			print("\t3. Define string variables with login information as follows:")
			print("\t\ta)username = <username>")
			print("\t\tb)password = <password>")
			print("\t4. Relaunch the program.")
			print("\nYour username and password will be automatically used as the default OpenSky Network login.")
			
			print("\nAlternatively, navigate to src/tabs/live.py, find this exception code, and change the defaults below this line.") """
			
			self.username.set("N/A")
			self.password.set("N/A")
		
		#self.api = OpenSkyApi(username = self.username.get(), password = self.password.get())
		#self.api = OpenSkyApi(username = login.username, password = login.password)

		self.df = pd.DataFrame()
		
		def set_api():
			print("[LIVE][set_api] Triggered...")
			#print(self.api._auth)
			try:
				self.api = OpenSkyApi(username = self.username.get(), password = self.password.get())
				_ = self.api.get_states().states[0]
			except AttributeError as e:
				print(e)
				self.button_get.child['state'] = 'disabled'
				self.login_labelframe['text'] = "OpenSky Network Login"
				
				messagebox.showerror(title="Error", message="Username or password is incorrect.")
				return
			
			self.login_labelframe['text'] = "OpenSky Network Login - [Currenly Logged In]"
			self.button_get.child['state'] = 'normal'
		
		
		#self.after(10000, set_api())
		
		
		
		# Login Container
		self.login_labelframe = sku.CustomLabelFrame(self, text="OpenSky Network Login")
		self.login_labelframe.grid(row=0, column=0, rowspan=1, columnspan=9, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		self.login_labelframe.grid_rowconfigure(0, weight=1)
		self.login_labelframe.grid_columnconfigure(0, weight=1)
		self.login_labelframe.grid_columnconfigure(1, weight=1)
		self.login_labelframe.grid_columnconfigure(2, weight=1)
		
		# Username
		self.username_labelframe = sku.CustomLabelFrame(self.login_labelframe, text="Username: ", labelanchor = "w")
		self.username_labelframe.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		self.username_labelframe.grid_rowconfigure(0, weight=1)
		self.username_labelframe.grid_columnconfigure(0, weight=1)
		self.username_labelframe['font'] = sku.FONT_NORM
		self.username_entry = sku.CustomEntry(self.username_labelframe, textvariable = self.username)
		self.username_entry.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		
		# Password
		self.password_labelframe = sku.CustomLabelFrame(self.login_labelframe, text="Password: ", labelanchor = "w")
		self.password_labelframe.grid(row=0, column=1, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		self.password_labelframe.grid_rowconfigure(0, weight=1)
		self.password_labelframe.grid_columnconfigure(0, weight=1)
		self.password_labelframe['font'] = sku.FONT_NORM
		self.password_entry = sku.CustomEntry(self.password_labelframe, text = self.password, show = "*")
		self.password_entry.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		# Sign In
		self.button_signin = sku.BorderButton(self.login_labelframe, button_text = 'Sign In', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: [self.username.set(self.username_entry.get()), self.password.set(self.password_entry.get()), set_api()])
		self.button_signin.grid(row = 0, column = 2, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		
		
		
		
		
		
		
		
		
		
		
		
		
		#print("[LIVE] PATH: {}".format(Path.cwd() / "src" / "assets" / "on_und.png"))
		# Switch Control
		self.is_on = False
		
		
		self.on = PhotoImage(file=Path.cwd() / "src" / "assets" / "on_und.png")
		self.off = PhotoImage(file=Path.cwd() / "src" / "assets" / "off_und.png")
		self.on.zoom(58, 24)
		self.off.zoom(58, 24)
		
		#self.on = PhotoImage(file=os.path.join(os.getcwd() + "/src/assets/on_und.png"))
		
		
		#self.off = PhotoImage(file=os.path.join(os.getcwd() + "/src/assets/off_und.png"))
		
		

		# Live Tracking
		self.labelframe_live = sku.CustomLabelFrame(self, text="Live Tracking")
		self.labelframe_live.grid(row=1, column=6, rowspan=1, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		self.labelframe_live.grid_rowconfigure(0, weight=1)
		for col in range(1):
			self.labelframe_live.grid_columnconfigure(col, weight=1)

		
		

		# Plot Frame A
		self.frame_a = sku.BorderFrame(self, background='#505050', border_color="green")
		self.frame_a.grid(row=0, column=9, rowspan=5, columnspan=5, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		self.frame_a.nametowidget('child').grid_columnconfigure(0, weight=1)  # , minsize = 100)
		self.frame_a.nametowidget('child').grid_rowconfigure(0, weight=1)
		self.frame_a.nametowidget('child').grid_rowconfigure(1, weight=0, minsize=50)

		# Plot Frame B
		self.frame_b = sku.BorderFrame(self, background='#505050', border_color="green")
		self.frame_b.grid(row=6, column=9, rowspan=7, columnspan=9, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		self.frame_b.nametowidget('child').grid_columnconfigure(0, weight=1)  # , minsize = 100)
		self.frame_b.nametowidget('child').grid_rowconfigure(0, weight=1)
		self.frame_b.nametowidget('child').grid_rowconfigure(1, weight=0, minsize=50)


		#Setup Live Plot A
		self.fig_a = Figure()
		#ax = fig.add_subplot(111)
		self.ax_a = self.fig_a.add_subplot(1,1,1, projection=crs.Mercator())
		self.ax_a.add_feature(cfeature.COASTLINE)
		self.ax_a.add_feature(cfeature.STATES)# Zoom in on the US by setting longitude/latitude parameters
		# ax.set_extent(
		# 	[
		# 		-135, # minimum latitude
		# 		-60, # min longitude
		# 		20, # max latitude
		# 		55 # max longitude
		# 	],
		# 	crs=crs.Mercator()
		# )


		#Setup Live Plot B
		self.fig_b = Figure()
		self.ax_b = self.fig_b.add_subplot(111)

		#Place graph A
		self.canvas_a = FigureCanvasTkAgg(self.fig_a, self.frame_a)
		self.canvas_a.draw()
		self.canvas_a.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW", padx=PADX_CONFIG, pady=(2, 0))

		self.toolbarFrame_a = tk.Frame(self.frame_a)
		self.toolbarFrame_a.grid(row=1,column=0, sticky = "NSEW", padx=PADX_CONFIG, pady=(0,2))
		self.toolbarFrame_a.grid_rowconfigure(0, weight = 1)
		self.toolbarFrame_a.grid_columnconfigure(0, weight = 1)
	
		self.toolbar_a = NavigationToolbar2Tk(self.canvas_a, self.toolbarFrame_a)
		self.toolbar_a.grid(row = 0, column = 0, sticky="NSEW")	


		#Place graph B
		self.canvas_b = FigureCanvasTkAgg(self.fig_b, self.frame_b)
		self.canvas_b.draw()
		self.canvas_b.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW", padx=PADX_CONFIG, pady=(2, 0))

		self.toolbarFrame_b = tk.Frame(self.frame_b)
		self.toolbarFrame_b.grid(row=1,column=0, sticky = "NSEW", padx=PADX_CONFIG, pady=(0,2))
		self.toolbarFrame_b.grid_rowconfigure(0, weight = 1)
		self.toolbarFrame_b.grid_columnconfigure(0, weight = 1)
	
		self.toolbar_b = NavigationToolbar2Tk(self.canvas_b, self.toolbarFrame_b)
		self.toolbar_b.grid(row = 0, column = 0, sticky="NSEW")	

		#_thread.start_new_thread(plotting_thread, (fig, ax))

		#plt.show()


		#live_data()
		
		
		
		






		# Get Aircraft Data
		self.button_get = sku.BorderButton(self, button_text = 'Get Live Aircraft Data', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: [get_states(self.scrolledtext_console, self.api)])
		self.button_get.grid(row = 1, column = 0, rowspan = 1, columnspan = 3, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		self.button_get.child['state'] = 'disabled'

		# Start Tracking
		self.button_track = sku.BorderButton(self, button_text = 'Start Tracking', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: [_thread.start_new_thread(plotting_thread, (self.fig_a, self.ax_a, self.fig_b, self.ax_b, self.scrolledtext_console))])
		self.button_track.grid(row = 1, column = 3, rowspan = 1, columnspan = 3, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		self.button_track.child['state'] = 'disabled'
		
		
		
		# Show ICAO24
		#button_keys = sku.BorderButton(self, button_text = 'Show ICAO24 Numbers', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: [json_parser.show_keys(scrolledtext_console, path)])
		#button_keys.grid(row = 2, column = 3, rowspan = 1, columnspan = 3, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)		
		

		switch(self.labelframe_live)


		
		
		
		
		
		
		
			
		#Console
		frame_console = sku.BorderFrame(self, background = '#505050', border_color = 'green')
		frame_console.grid(row = 3, column = 0, rowspan = 9, columnspan = 9, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		frame_console.nametowidget('child').grid_rowconfigure(0, weight = 1)
		frame_console.nametowidget('child').grid_columnconfigure(0, weight = 1)
		
		self.scrolledtext_console = sku.CustomScrolledText(frame_console.nametowidget('child'))
		self.scrolledtext_console.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		#self.scrolledtext_console['state'] = 'disabled'
		
		
		# Clear Console
		button_clear = sku.BorderButton(self, button_text = 'Clear Console', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: self.scrolledtext_console.delete(1.0, END))
		button_clear.grid(row = 12, column = 0, rowspan = 1, columnspan = 3, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
	

'''
if __name__ == "__main__":
	new = tk.Tk()
	page = Parse(new)
	page.pack(fill="both", expand=True)
	new.mainloop()
	'''