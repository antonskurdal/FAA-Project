#!/usr/bin/env python

"""This file controls the Injection Tab.

	Injection Tab description.
"""

import tkinter as tk
from tkinter import *
from pathlib import Path
from tkinter import messagebox
import pandas as pd
from dataclasses import dataclass
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

import util.sku_widgets as sku
import util.grapher as grapher
import util.stat_calc as stat_calc

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


# Padding
PADX_CONFIG = (2, 2)
PADY_CONFIG = (2, 2)

# Inject Tab
class Inject(tk.Frame):

	def __init__(self, parent, controller, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs, bg = "black")#bg = sku.FRAME_BACKGROUND)
		self.controller = controller
		
		
		###################
		# Layout Controls #
		###################
		for row in range(9):
			self.grid_rowconfigure(row, weight=1)
		for col in range(18):
			self.grid_columnconfigure(col, weight=1)

		for row in range(9):
			self.grid_rowconfigure(row, weight=0, minsize=100)
		for col in range(18):
			self.grid_columnconfigure(row, weight=0, minsize=100)
		
		
		##############################
		# 		DATA CONTAINERS		 #
		##############################
		@dataclass
		class CurrentGraph:
			
			def __init__(self, base, current, xs_colname, ys_colname):
				
				self._base = base
				self._current = current
				
				self._xs_colname = xs_colname
				self._ys_colname = ys_colname
			
			#base
			@property
			def base(self) -> object:
				#I'm the 'base' property.
				return self._base

			@base.setter
			def base(self, value: object) -> object:
				self._base = value

			@base.deleter
			def base(self):
				del self._base	
			
			
			#current
			@property
			def current(self) -> object:
				#I'm the 'current' property.
				return self._current
			
			@current.setter
			def current(self, value: object) -> object:
				self._current = value
			
			@current.deleter
			def current(self):
				del self._current
			
			
			#xs_colname
			@property
			def xs_colname(self) -> object:
				#I'm the 'xs_colname' property.
				return self._xs_colname
			
			@xs_colname.setter
			def xs_colname(self, value: object) -> object:
				self._xs_colname = value
			
			@xs_colname.deleter
			def xs_colname(self):
				del self._xs_colname	
			
			
			#ys_colname
			@property
			def ys_colname(self) -> object:
				#I'm the 'ys_colname' property.
				return self._ys_colname
			
			@ys_colname.setter
			def ys_colname(self, value: object) -> object:
				self._ys_colname = value
			
			@ys_colname.deleter
			def ys_colname(self):
				del self._ys_colname
		
		
		##############################
		# 		CLASS METHODS		 #
		##############################

		def populate_listbox(listbox, df, sel=None):
			
			#print("[INJECT][POPULATE_LISTBOX] LISTBOX SIZE: {}".format(listbox.size()))
			if(listbox.size() > 0):
				listbox.delete(0, tk.END)
			
			
			# Allow rows with one or less unique value to appear in the selection listboxes
			allow_generic = False
			#print("[INJECT][POPULATE_LISTBOX] DF COLUMNS: {}".format(df.columns))
			#print(df['vertrate'])
			
			for col in df.columns:
				#print("[INJECT][POPULATE_LISTBOX] COL: {}, type={}".format(col, df.dtypes[col]))
				if(df.dtypes[col] == 'object'):
					#print("OBJECT\n")
					pass
				elif(len(df.dtypes[col]) > 1):
					#print("MORE THAN 1 DTYPE\n")
					pass
				elif(len(df[col].unique()) <= 1 and allow_generic == False):
					pass
				else:
					listbox.insert(tk.END, col)
			
			listbox.insert(tk.END, "index")
			
			if(listbox.size() > sel):
				listbox.selection_clear(0, tk.END)
				listbox.select_set(sel)
				listbox.activate(sel)
				listbox.update()
			else:
				listbox.selection_clear(0, tk.END)
				listbox.select_set(0)
				listbox.activate(0)
				listbox.update()
		
		def file_browse(directory, var):
			
			#print("[INJECT][FILE_BROWSE] DIRECTORY: {}".format(directory))
			
			file = Path(filedialog.askopenfilename(
				filetypes = [('All files', '.*'), ('CSV files', '.csv'), ('Apache Parquet files', '.parquet')], 
				title = "Dataset Selection", 
				initialdir = directory))
			
			#print("[INJECT][FILE_BROWSE] FILE NAME: {} (type = {})".format(file.name, type(file)))
			
			#if file is None: #askopenfilename return `None` if dialog closed with "cancel".
			if(file.name == ""): #askopenfilename return `None` if dialog closed with "cancel".
				messagebox.showwarning(title="Warning", message="No file selected")
				return
			else:
				var.set(str(file.name))
				file_load(file)
		
		def file_load(file):
			#print("[INJECT][FILE_LOAD] FILE: {} (type = {})".format(file.name, type(file)))
			
			# Destroy any existing plots when loading a new file
			for child in frame_plot.nametowidget('child').winfo_children():
				child.destroy()
			
			
			# Check file name and load 
			if(file.suffix == ".csv"):
				base_data = pd.read_csv(file)
			elif(file.suffix == ".parquet"):
				base_data = pd.read_parquet(file)
			elif(file.name == ""):
				messagebox.showwarning(title="Warning", message="No file selected")
				return
			else:
				messagebox.showerror(title="Error", message="Invalid file extension. Must be '.csv' or '.parquet'")
				return
			
			# Check if label column exists (normal, dropout, noise, etc.)
			if 'taxonomy' not in base_data.columns:
				base_data.insert(1, 'taxonomy', 'normal')
			
			##########################################################################
			base_data = remove_epoch(base_data)
			##########################################################################
			
			# Check if lastcontact column exists (normal, dropout, noise, etc.)
			if 'lastcontact' not in base_data.columns:
				button_stats_apply.child['state'] = 'disabled'
			else:
				button_stats_apply.child['state'] = 'normal'
			
			
			self.filecontroller_main.label.child_text.set(file.name)
			self.filecontroller_save_current.label.child_text.set(file.stem + "_modified")
			
			
			
			populate_listbox(listbox_xs, base_data, 0)
			populate_listbox(listbox_ys, base_data, 2)
			
			#global obj
			self.obj = CurrentGraph(base_data.copy(deep=True), base_data.copy(deep=True), listbox_xs.get(
				listbox_xs.curselection()), listbox_ys.get(listbox_ys.curselection()))
			
			self.DATA_DIR = file.parent
			#print("[INJECT][FILE_LOAD] DATA_DIR: {} (type = {})".format(self.DATA_DIR, type(self.DATA_DIR)))
			self.FILE = file
			
			
			#print("[INJECT][FILE_LOAD] FILE: {} (type = {})".format(self.FILE.stem, type(self.FILE)))
			
			sel_changed('<<ListboxSelect>>')


		def remove_epoch(df):
			
			# Dropout Score
			if("epoch_time" in df.columns):
				df['epoch_time'] = df['time']
			else:
				df.insert(1, 'epoch_time', df['time'])
			
			
			epoch = df['time'].iloc[0]
			print(epoch)
			df['time'] = df['time'] - epoch
			print(df.head())
			print(df.columns)
			return df

		###############################
		def tag_attacks():
			return
			# BROKEN! DOES NOT WORK WITH THE NEW 'INSERT POINTS' FUNCTIONALITY IN THE POLYGON INTERACTOR
			
			
			for i in range(len(self.obj.base)):

				if((i not in self.obj.base.index) | (i not in self.obj.current.index)):
					# print("Skipping " + str(i))
					continue

				if((self.obj.base.at[i, self.obj.xs_colname] != self.obj.current.at[i, self.obj.xs_colname]) or (self.obj.base.at[i, self.obj.ys_colname] != self.obj.current.at[i, self.obj.ys_colname])):

					self.obj.current.at[i, 'taxonomy'] = 'attack'

		################################
		def file_save(var, data):
			
			
			directory = self.DATA_DIR
			
			
			f = filedialog.asksaveasfile(
				filetypes = [('Apache Parquet files', '.parquet'), ('CSV files', '.csv')], 
				mode='w', defaultextension=".parquet", 
				initialfile = var.get(), 
				initialdir = directory
				)
			if f is None: #asksaveasfile return `None` if dialog closed with "cancel".
				return

			data.to_csv(f, index = False, line_terminator = '\n')
			f.close()
		
		def sel_changed(event):
			#Set object parameters
			#self.obj.current = self.obj.base.copy(deep=True)
			self.obj.xs_colname = listbox_xs.get(listbox_xs.curselection())
			self.obj.ys_colname = listbox_ys.get(listbox_ys.curselection())
			
			#print("[INJECT][FILE_BROWSE] DIRECTORY: {}".format(directory))
			#print("[INJECT][sel_changed] xs_colname: {}".format(self.obj.xs_colname))
			
			
			# Create array for slider
			if(self.obj.xs_colname == "index"):
				arr = self.obj.current.index
			else:
				print("[INJECT][sel_changed] xs not 'index'")
				arr = self.obj.current[self.obj.xs_colname]
			
			# Set min/max
			slider_min = min(arr.dropna())
			slider_max = max(arr.dropna())
			
			# Widen bounds by 0.001 on either side for floating point precision error tolerance
			# print("BEFORE:")
			# print("slider_min: {}".format(slider_min))
			# print("slider_max: {}".format(slider_max))
			slider_min -= 0.001
			slider_max += 0.001
			
			# Disable buttons if y-axis is 'index'
			if(self.obj.ys_colname == "index"):
				button_fit.child['state'] = 'disabled'
				button_drop.child['state'] = 'disabled'
				button_noise.child['state'] = 'disabled'
				button_percent.child['state'] = 'disabled'
				button_num.child['state'] = 'disabled'
			else:
				print("[INJECT][sel_changed] ys not 'index'")
				button_fit.child['state'] = 'normal'
				button_drop.child['state'] = 'normal'
				button_noise.child['state'] = 'normal'
				button_percent.child['state'] = 'normal'
				button_num.child['state'] = 'normal'
			
			# print("AFTER:")
			# print("slider_min: {}".format(slider_min))
			# print("slider_max: {}".format(slider_max))
			
			
			
			# sku.get_attributes(self.slider)
			# self.slider.__setattr__('min_val', slider_min)
			# self.slider.__setattr__('max_val', slider_max)
			# self.slider.__setattr__('init_lis', [slider_min, slider_max])
			
			
			# print("SLIDER MIN: {}".format(self.slider.min_val))
			
			# self.slider.update_idletasks()
			
			# #self.slider.grid_forget()
			# self.slider.grid(row=0, column=0, rowspan=1, columnspan=3,sticky="NSEW", padx=(3,2), pady=PADY_CONFIG)
			
			
			
			#Create slider
			self.slider = sku.LiSlider(
				labelframe_slider, 
				width=650, 
				height=60, 
				min_val=slider_min, 
				max_val=slider_max, 
				init_lis=[slider_min, slider_max], 
				show_value=True
				)
			
			
			# """ #Create slider
			# self.slider = sku.LiSlider(
			# 	labelframe_slider, 
			# 	width=500, 
			# 	height=60, 
			# 	min_val=min(self.obj.current[self.obj.xs_colname]), 
			# 	max_val=max(self.obj.current[self.obj.xs_colname]), 
			# 	init_lis=[min(self.obj.current[self.obj.xs_colname]), max(self.obj.current[self.obj.xs_colname])], 
			# 	show_value=True
			# 	) """
			
			self.slider.grid(row=0, column=0, rowspan=1, columnspan=3,sticky="NSEW", padx=(3,2), pady=PADY_CONFIG)
			self.slider.canv['bg'] = sku.SCALE_BACKGROUND
			self.slider.canv.master['bg'] = sku.SCALE_BACKGROUND
			self.slider.canv['highlightthickness'] = 0
			self.slider.canv.master['highlightthickness'] = 2
			self.slider.canv.master['highlightbackground'] = sku.SCALE_HIGHLIGHTBACKGROUND
			
			grapher.plotInteractiveLine(frame_plot.nametowidget('child'), self.obj, self.slider)
			
		
		
		########################
		# Modification Methods #
		########################
		def get_fit(master, degree):

			from numpy import array as nparray
			from numpy import polyfit as nppolyfit
			from numpy import poly1d as nppoly1d

			x = nparray(self.obj.current[self.obj.xs_colname])
			y = nparray(self.obj.current[self.obj.ys_colname])

			map(float, x)
			map(float, y)

			weights = nppolyfit(x, y, degree)
			model = nppoly1d(weights)

			self.obj.current[self.obj.xs_colname] = x
			self.obj.current[self.obj.ys_colname] = model(x)
			#grapher.plotInteractivePolygon(master, self.obj)
			grapher.plotInteractiveLine(master, self.obj)
		
		def drop_data(master, bounds):
			
			print(bounds)
			print("[INJECT][drop_data] Bounds:{}".format(bounds))
			
			df = self.obj.current.copy()
			# Create array
			if(self.obj.xs_colname == "index"):
				arr = df.index
			else:
				print("[INJECT][drop_data] xs not 'index'")
				arr = df[self.obj.xs_colname]
			
			
			lower = df[arr <= bounds[0]]
			higher = df[arr >= bounds[1]]
			df = pd.concat([lower, higher])
			print("[INJECT][drop_data] df:\n{}\n".format(df))
			
			self.obj.current = df
			
			
			
			# t = data.index
			# test_low = data[t <= bounds[0]]
			# test_high = data[t >= bounds[1]]
			# test_data = pd.concat([test_low, test_high])
			
			# print("[INJECT][drop_data] test:\n{}\n".format(test_data))
			
			
			
			
			
			""" data = self.obj.current
			x = self.obj.xs_colname
			y = self.obj.ys_colname
			
			
			

			
			low = data[data[x] < bounds[0]]
			
			high = data[data[x] > bounds[1]]
			
			data = pd.concat([low, high])
			
			self.obj.current = data """
			#grapher.plotInteractivePolygon(master, self.obj)
			
			#grapher.plotInteractiveLine(master, self.obj, self.slider)
			sel_changed('<<ListboxSelect>>')
			
			#sel_changed(self)
			
		def add_noise(master, bounds, percent):
			
			from random import randint
			
			print(bounds)
			print(percent)
			
			x = self.obj.xs_colname
			y = self.obj.ys_colname
			data = self.obj.current
			
			# Get data lower than bounds
			low = data[data[x] < bounds[0]]
			
			# Get data higher than bounds
			high = data[data[x] > bounds[1]]
			
			# Remove lower than bounds
			mid = data[data[x] >= bounds[0]]
			
			# Remove data higher than bounds
			mid = mid[mid[x] <= bounds[1]]
			
			mid = mid.reset_index(drop = True)
			
			# Modify
			for i in range(len(mid[y])):
				
				rand = randint(percent*-1, percent)
				
				if (rand != 0):
					rand_pct = rand/100
				else:
					rand_pct = 0
				
				mid.at[i, y] = mid.at[i, y] + (mid.at[i, y] * rand_pct)
			
			
			# Concat data frames
			data = pd.concat([low, mid, high])
			data = data.reset_index(drop = True)
			
			self.obj.current = data
			#grapher.plotInteractivePolygon(master, self.obj)
			grapher.plotInteractiveLine(master, self.obj)
			
		def add_percent(master, bounds, percent):
			x = self.obj.xs_colname
			y = self.obj.ys_colname
			data = self.obj.current
			
			# Get data lower than bounds
			low = data[data[x] < bounds[0]]
			
			# Get data higher than bounds
			high = data[data[x] > bounds[1]]
			
			# Remove lower than bounds
			mid = data[data[x] >= bounds[0]]
			
			# Remove data higher than bounds
			mid = mid[mid[x] <= bounds[1]]
			
			mid = mid.reset_index(drop = True)
			
			# Modify
			for i in range(len(mid[y])):
				
				mid.at[i, y] = mid.at[i, y] + (mid.at[i, y] * percent/100)
			
			
			# Concat data frames
			data = pd.concat([low, mid, high])
			data = data.reset_index(drop = True)
			
			self.obj.current = data
			#grapher.plotInteractivePolygon(master, self.obj)
			grapher.plotInteractiveLine(master, self.obj)
		
		def add_num(master, bounds, number, do_rolling):
			
			try:
				number = float(number)
			except ValueError:
				messagebox.showerror(title = "Invalid Input", message = "Input for 'Add Numeric' must be a number.")
				return
			
			x = self.obj.xs_colname
			y = self.obj.ys_colname
			data = self.obj.current
			
			# Get data lower than bounds
			low = data[data[x] < bounds[0]]
			
			# Get data higher than bounds
			high = data[data[x] > bounds[1]]
			
			# Remove lower than bounds
			mid = data[data[x] >= bounds[0]]
			
			# Remove data higher than bounds
			mid = mid[mid[x] <= bounds[1]]
			
			mid = mid.reset_index(drop = True)
			
			# Modify
			for i in range(len(mid[y])):
				
				mid.at[i, y] = mid.at[i, y] + number
				
				if(do_rolling == True):
					number += number
					print(number)
			
			
			# Concat data frames
			data = pd.concat([low, mid, high])
			data = data.reset_index(drop = True)
			
			self.obj.current = data
			#grapher.plotInteractivePolygon(master, self.obj)
			grapher.plotInteractiveLine(master, self.obj)
					
		
		
		
		def switch():
		
			# Determine if switch is on or off
			if self.is_on:
				
				labelframe_num_switch = sku.CustomLabelFrame(labelframe_num, text = "Rolling", labelanchor = 'n')
				labelframe_num_switch.grid(row = 0, column = 2, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
				labelframe_num_switch.grid_anchor('center')
				
				switch_num = sku.BorderButton(labelframe_num_switch, button_image = self.off, button_command = switch, button_activebackground = '#404040')
				switch_num.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
				
				# label_num.config(label_num.child_text.set("The Switch is Off"), bg = "red")
				self.is_on = False
			else:
				
				labelframe_num_switch = sku.CustomLabelFrame(labelframe_num, text = "Rolling", labelanchor = 'n')
				labelframe_num_switch.grid(row = 0, column = 2, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
				labelframe_num_switch.grid_anchor('center')
				
				switch_num = sku.BorderButton(labelframe_num_switch, button_image = self.on, button_command = switch, button_activebackground = '#404040')
				switch_num.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
				
				# label_num.config(label_num.child_text.set("The Switch is On"), bg = "green")
				self.is_on = True		
		
		
		
		######################
		# 		WIDGETS		 #
		######################
		
		##############
		# Left Third #
		##############
		
		# Input File Loader
		self.filecontroller_main = sku.FileController(self, text="Load CSV/Parquet File", label_text="", button_text="Browse", button_command=
		lambda: [
			file_browse(self.DATA_DIR, self.filecontroller_main.label.child_text), 
			sku.flash_zone(self.filecontroller_main.label, 'bg', 'green'), 
			#self.filecontroller_save_current.label.child_text.set(self.FILE.stem + "_injected" + self.FILE.suffix), 
			sku.flash_zone(self.filecontroller_save_current.label, 'bg', 'green')
			])
		self.filecontroller_main.grid(row=0, column=0, rowspan=1, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		# X-Axis Selection
		labelframe_listbox_xs = sku.CustomLabelFrame(self, text="X-Axis")
		labelframe_listbox_xs.grid(row=1, column=0, rowspan=3, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		labelframe_listbox_xs.grid_rowconfigure(0, weight=1)
		labelframe_listbox_xs.grid_columnconfigure(0, weight=1)
		listbox_xs = sku.CustomListbox(labelframe_listbox_xs, selectmode='SINGLE', exportselection=0)
		listbox_xs.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		listbox_xs.bind("<<ListboxSelect>>", sel_changed)

		# Y-Axis Selection
		labelframe_listbox_ys = sku.CustomLabelFrame(self, text="Y-Axis")
		labelframe_listbox_ys.grid(row=1, column=3, rowspan=3, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		labelframe_listbox_ys.grid_rowconfigure(0, weight=1)
		labelframe_listbox_ys.grid_columnconfigure(0, weight=1)
		listbox_ys = sku.CustomListbox(labelframe_listbox_ys, selectmode='SINGLE', exportselection=0)
		listbox_ys.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		listbox_ys.bind("<<ListboxSelect>>", sel_changed)
		
		# Plot
		button_plot = sku.BorderButton(self, button_text='Plot Columns', button_activebackground='green', button_command=lambda: [sel_changed('<<ListboxSelect>>')])#grapher.plotInteractiveLine(frame_plot.nametowidget('child'), self.obj, self.slider)])
		button_plot.grid(row=4, column=0, rowspan=1, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		
		# Reset Plot
		button_reset = sku.BorderButton(self, button_text='Reset Plot', button_activebackground='green', button_command=lambda: [self.obj.__setattr__('current', self.obj.base.copy(deep = True)), sel_changed('<<ListboxSelect>>')])
		button_reset.grid(row=4, column=3, rowspan=1, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		# Print Base Data
		button_print_base = sku.BorderButton(self, button_text='Show Base Data', button_activebackground='green', button_command=lambda: [print(self.obj.base[[self.obj.xs_colname, self.obj.ys_colname, 'taxonomy']]), print(self.obj.base.columns)])
		button_print_base.grid(row=5, column=0, rowspan=1, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		# Print Current Data
		button_print_current = sku.BorderButton(self, button_text='Show Current Data', button_activebackground='green', button_command=lambda: [tag_attacks(), print(self.obj.current[[self.obj.xs_colname, self.obj.ys_colname, 'taxonomy']]), print(self.obj.current.columns)])
		button_print_current.grid(row=5, column=3, rowspan=1, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		# Save Modified
		self.filecontroller_save_current = sku.FileController(self, text="Save Modified CSV File", label_text="", button_text="Save", button_command=lambda: [tag_attacks(), file_save(self.filecontroller_save_current.label.child_text, self.obj.current), sku.flash_zone(self.filecontroller_save_current.label, 'bg', 'green')])
		self.filecontroller_save_current.grid(row=6, column=0, rowspan=1, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		
		################
		# Middle Third #
		################

		# Plot Frame A
		frame_plot = sku.BorderFrame(self, background='#505050', border_color="#404040")
		frame_plot.grid(row=0, column=6, rowspan=6, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		frame_plot.nametowidget('child').grid_columnconfigure(0, weight=1)  # , minsize = 100)
		frame_plot.nametowidget('child').grid_rowconfigure(0, weight=1)
		frame_plot.nametowidget('child').grid_rowconfigure(1, weight=0, minsize=50)
		#self.grid_propagate(False)
		#frame_plot.grid_propagate(False)
		
		# Double Slider Parent Widget
		labelframe_slider = sku.CustomLabelFrame(self, text="X-Axis Modification Range Selection (Inclusive)")
		labelframe_slider.grid(row=6, column=6, rowspan=1, columnspan=6,sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		labelframe_slider.grid_rowconfigure(0, weight=1)
		for col in range(3):
			labelframe_slider.grid_columnconfigure(col, weight=1)
		
		#Create slider
		self.slider = sku.LiSlider(labelframe_slider, width=500, height=60, min_val=0, max_val=100, init_lis=[0, 100], show_value=True)	
		self.slider.grid(row=0, column=0, rowspan=1, columnspan=3,sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		self.slider.canv['bg'] = sku.SCALE_BACKGROUND
		self.slider.canv.master['bg'] = sku.SCALE_BACKGROUND
		self.slider.canv['highlightthickness'] = 0
		self.slider.canv.master['highlightthickness'] = 2
		self.slider.canv.master['highlightbackground'] = sku.SCALE_HIGHLIGHTBACKGROUND
		
		
		###############
		# Right Third #
		###############

		# Polynomial fit
		labelframe_degree = sku.CustomLabelFrame(self, text="Polynomial Fit Degree")
		labelframe_degree.grid(row=0, column=12, rowspan=1, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		labelframe_degree.grid_rowconfigure(0, weight=1)
		for col in range(5):
			labelframe_degree.grid_columnconfigure(col, weight=1)

		button_fit = sku.BorderButton(labelframe_degree, button_text='Fit Data', button_activebackground='green', button_command=lambda: get_fit(frame_plot.nametowidget('child'), scale_degree.get()))
		button_fit.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		scale_degree = sku.CustomScale(labelframe_degree, from_=1.0, to=15.0, resolution=1, tickinterval=1, showvalue=True)
		scale_degree.grid(row=0, column=1, rowspan=1, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		scale_degree.set(9.0)

		# Drop Data
		button_drop = sku.BorderButton(labelframe_degree, button_text='Drop', button_activebackground='green', button_command=lambda: [drop_data(frame_plot.nametowidget('child'), self.slider.getValues())])
		button_drop.grid(row=0, column=4, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		# Add Noise
		labelframe_noise = sku.CustomLabelFrame(self, text="Randomized Jamming/Noise Injection Maximum Percentage")
		labelframe_noise.grid(row=1, column=12, rowspan=1, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		labelframe_noise.grid_rowconfigure(0, weight=1)
		for col in range(3):
			labelframe_noise.grid_columnconfigure(col, weight=1)

		button_noise = sku.BorderButton(labelframe_noise, button_text='Add Noise', button_activebackground='green', button_command=lambda: [add_noise(frame_plot.nametowidget('child'), self.slider.getValues(), scale_noise.get())])  # lambda: print(slider.getValues()))
		button_noise.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		scale_noise = sku.CustomScale(labelframe_noise, from_=0.0, to=20, resolution=1, tickinterval=2, showvalue=True)
		scale_noise.grid(row=0, column=1, rowspan=1, columnspan=2, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		scale_noise.set(50.0)

		# Percentage Modification
		labelframe_percent = sku.CustomLabelFrame(self, text="Modify by Percentage")
		labelframe_percent.grid(row=2, column=12, rowspan=1, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		labelframe_percent.grid_rowconfigure(0, weight=1)
		for col in range(3):
			labelframe_percent.grid_columnconfigure(col, weight=1)

		button_percent = sku.BorderButton(labelframe_percent, button_text='Add Percent', button_activebackground='green', button_command=lambda: [add_percent(frame_plot.nametowidget('child'), self.slider.getValues(), scale_percent.get())])  # lambda: print(slider.getValues()))
		button_percent.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		scale_percent = sku.CustomScale(labelframe_percent, from_=-100.0, to=100.0, resolution=1, tickinterval=20, showvalue=True)
		scale_percent.grid(row=0, column=1, rowspan=1, columnspan=2, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		scale_percent.set(50.0)

		# Numeric Modification
		labelframe_num = sku.CustomLabelFrame(self, text="Numeric Modification")
		labelframe_num.grid(row=3, column=12, rowspan=1, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		labelframe_num.grid_rowconfigure(0, weight=1)
		for col in range(3):
			labelframe_num.grid_columnconfigure(col, weight=1)

		button_num = sku.BorderButton(labelframe_num, button_text='Add Numeric', button_activebackground='green', button_command=lambda: [add_num(frame_plot.nametowidget('child'), self.slider.getValues(), entry_num_var.get(), self.is_on)])  # lambda: print(slider.getValues()))
		button_num.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		entry_num_var = tk.StringVar()
		entry_num = sku.CustomEntry(labelframe_num, textvariable=entry_num_var)
		entry_num.grid(row=0, column=1, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		entry_num_var.set(5)

		# Switch Control
		self.is_on = True
		self.on = PhotoImage(file=Path.cwd() / "src" / "assets" / "on_und.png")
		self.off = PhotoImage(file=Path.cwd() / "src" / "assets" / "off_und.png")
		self.on.zoom(58, 24)
		self.off.zoom(58, 24)
		switch()

		
		# Switch Assets
		switch_on = PhotoImage(file=Path.cwd() / "src" / "assets" / "on_und.png")
		switch_off = PhotoImage(file=Path.cwd() / "src" / "assets" / "off_und.png")
		
		
		
		def sel():
			selection = "Sel: " + str(radio_string.get())
			#label.config(text = selection)
		radio_string = tk.StringVar()
		
		# Apply Taxonomy
		labelframe_taxonomy = sku.CustomLabelFrame(self, text="Apply Taxonomy to Current Range [2]")
		labelframe_taxonomy.grid(row=4, column=12, rowspan=1, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		labelframe_taxonomy.grid_rowconfigure(0, weight=1)
		for row in range(2):
			labelframe_taxonomy.grid_rowconfigure(row, weight = 1)
		
		for col in range(5):
			labelframe_taxonomy.grid_columnconfigure(col, weight=1)
		labelframe_taxonomy.grid_columnconfigure(3, weight=0)
		#labelframe_taxonomy.grid_propagate(False)
		
		radio_on = ImageTk.PhotoImage(file=Path.cwd() / "src" / "assets" / "radio_selected.png")
		radio_off = ImageTk.PhotoImage(file=Path.cwd() / "src" / "assets" / "radio_unselected.png")
		radio_test = ImageTk.PhotoImage(file=Path.cwd() / "src" / "assets" / "radio_test.png")
		radio_test2 = ImageTk.PhotoImage(file=Path.cwd() / "src" / "assets" / "radio_test.png")
		
		
		# Normal
		radio_normal = sku.BorderRadiobutton(labelframe_taxonomy, activebordercolor = "#009A44", text = "normal", variable = radio_string, command = sel, value = "normal", indicator = 0)
		radio_normal.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		#sku.get_attributes(radio_normal.radiobutton)
		
		# Dropout
		radio_dropout = sku.BorderRadiobutton(labelframe_taxonomy, activebordercolor = "#009A44", text = "dropout", variable = radio_string, command = sel, value = "dropout", indicator = 0)
		radio_dropout.grid(row=1, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		# Noise
		radio_noise = sku.BorderRadiobutton(labelframe_taxonomy, activebordercolor = "#009A44", text = "noise", variable = radio_string, command = sel, value = "noise", indicator = 0)
		radio_noise.grid(row=0, column=1, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		# Jammed
		radio_jammed = sku.BorderRadiobutton(labelframe_taxonomy, activebordercolor = "#009A44", text = "jammed", variable = radio_string, command = sel, value = "jammed", indicator = 0)
		radio_jammed.grid(row=1, column=1, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		# Spoofed
		radio_spoofed = sku.BorderRadiobutton(labelframe_taxonomy, activebordercolor = "#009A44", text = "spoofed", variable = radio_string, command = sel, value = "spoofed", indicator = 0)
		radio_spoofed.grid(row=0, column=2, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		# Erroneous
		radio_erroneous = sku.BorderRadiobutton(labelframe_taxonomy, activebordercolor = "#009A44", text = "erroneous", variable = radio_string, command = sel, value = "erroneous", indicator = 0)
		radio_erroneous.grid(row=1, column=2, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		
		# Separator
		sep_taxonomy = tk.Frame(labelframe_taxonomy, bg = '#AEAEAE', width = 2)
		sep_taxonomy.grid(row=0, column=3, rowspan=2, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		""" # Label
		label = sku.CustomLabel(labelframe_taxonomy, width = 20)
		label.grid(row=0, column=4, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		label.grid_propagate(False) """
		
		# Apply
		button_taxonomy_apply = sku.BorderButton(labelframe_taxonomy, button_text="Apply", button_activebackground='green', button_command=lambda: 
		[
			setattr(self.obj, 'current', stat_calc.apply_taxonomy(self.obj.current, self.obj.xs_colname, self.slider.getValues(), radio_string.get())),
		])
		button_taxonomy_apply.grid(row=0, column=4, rowspan=2, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		# Set default value
		radio_dropout.radiobutton.invoke()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		def calc_statistics():
			
			# Set booleans
			current_dataset = switch_dataset.get_state()
			dropout_length = switch_dropout_length.get_state()
			stddev_zscore = switch_dropout_zscore.get_state()
			sma = switch_dropout_sma.get_state()
			snr = switch_dropout_snr.get_state()
			modedev_zscore = switch_dropout_modedev.get_state()
			constants = switch_dropout_constants.get_state()
			
			# Return if nothing is true
			if not any(v == True for v in [dropout_length, stddev_zscore, sma, snr, modedev_zscore, constants]):
				#print("NONE TRUE")
				return
			
			# Calculate dropout_length if any others are true
			if any(v == True for v in [stddev_zscore, sma, snr, modedev_zscore, constants]):
				#print("TRUE")
				dropout_length = True
				#return
			
			# Assign dataset
			if(switch_dataset.get_state()):
				#info += "- [Dataset]: current\n"
				df = self.obj.current
			else:
				#info += "- [Dataset]: base\n"
				df = self.obj.base
			
			
			# Dropout Length
			if(dropout_length):
				stat_calc.dropouts(df)
			
			"""CONSTANTS?"""
			
			# Standard Deviation Z-Score
			if(stddev_zscore):
				stat_calc.stdev_zscore(df)
			
			# Simple Moving Average (window = 25)
			if(switch_dropout_sma):
				stat_calc.simple_moving_average(df, 10)
			
			# Signal to Noise Ratio (Rolling)
			if(switch_dropout_snr):
				stat_calc.snr_rolling(df, 0, 0)
			
			# Mode Deviation Z-Score
			if(switch_dropout_modedev):
				stat_calc.mode_deviation(df)
			
			return
		
		
		# Calculate Statistic Columns
		labelframe_stats = sku.CustomLabelFrame(self, text="Create Statistic Columns")
		labelframe_stats.grid(row=5, column=12, rowspan=2, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		for row in range(2):
			labelframe_stats.grid_rowconfigure(row, weight=1)
		for col in range(5):
			labelframe_stats.grid_columnconfigure(col, weight=1)
		labelframe_stats.grid_columnconfigure(3, weight=0)
		labelframe_stats.grid_propagate(False)
		# Dropout Length Switch
		switch_dropout_length = sku.CustomSwitch(labelframe_stats, text="Dropout Length\n(seconds)", textanchor = "n", on_image = switch_on, off_image=switch_off, init_state = True)
		switch_dropout_length.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		# Dropout Z-Score Switch
		switch_dropout_zscore = sku.CustomSwitch(labelframe_stats, text="Std Deviation\nZ-Score", textanchor = "n", on_image = switch_on, off_image=switch_off, init_state = True)
		switch_dropout_zscore.grid(row=0, column=1, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		# Dropout Moving Average
		switch_dropout_sma = sku.CustomSwitch(labelframe_stats, text="Simple Moving\nAverage\n(w=10)", textanchor = "n", on_image = switch_on, off_image=switch_off, init_state = True)
		switch_dropout_sma.grid(row=0, column=2, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		# Dropout Signal to Noise Ratio
		switch_dropout_snr = sku.CustomSwitch(labelframe_stats, text="Signal to Noise Ratio\n(rolling)", textanchor = "n", on_image = switch_on, off_image=switch_off, init_state = True)
		switch_dropout_snr.grid(row=1, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		# Dropout Mode Deviation
		switch_dropout_modedev = sku.CustomSwitch(labelframe_stats, text="Mode Deviation\nZ-Score", textanchor = "n", on_image = switch_on, off_image=switch_off, init_state = True)
		switch_dropout_modedev.grid(row=1, column=1, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		# Mean, Mode, Std, Mode Dev
		switch_dropout_constants = sku.CustomSwitch(labelframe_stats, text="Mean, Mode, Std, Mode Dev\n(Constants)", textanchor = "n", on_image = switch_on, off_image=switch_off, init_state = True)
		#switch_dropout_constants.grid(row=1, column=2, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		""" # Dropout Linear Regression
		switch_dropout_linearregression = sku.CustomSwitch(labelframe_stats, text="Linear Regression", textanchor = "n", on_image = switch_on, off_image=switch_off, init_state = True)
		switch_dropout_linearregression.grid(row=1, column=2, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG) """
		
		
		
		
		# Separator
		sep_stats = tk.Frame(labelframe_stats, bg = '#AEAEAE', width = 2)
		sep_stats.grid(row=0, column=3, rowspan=2, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		# Modify Current Data Switch
		switch_dataset = sku.CustomSwitch(labelframe_stats, text="Modify Current\n Dataset", textanchor = "n", on_image = switch_on, off_image=switch_off, init_state = True)
		switch_dataset.grid(row=0, column=4, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		# Apply Stats
		button_stats_apply = sku.BorderButton(labelframe_stats, button_text="Apply", button_activebackground='green', button_command=lambda: 
		[
			calc_statistics(),
		])
		button_stats_apply.grid(row=1, column=4, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		
		
		
		# Autotag
		button_autotag = sku.BorderButton(self, button_text="Autotag", button_activebackground='green', button_command=lambda: 
		[
			stat_calc.score(self.obj.current),
			stat_calc.autotag(self.obj.current)
		])
		button_autotag.grid(row=7, column=12, rowspan=1, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		
		
		
		
		
		
		
		
		
		
		
		# Banner Image
		img = Image.open(Path.cwd() / "src" / "assets" / "und_banner.png")
		self.photoImg = ImageTk.PhotoImage(img)

		# Banner
		frame_banner = sku.BorderFrame(self, background='#505050', border_color="green")
		frame_banner.grid(row=7, column=0, rowspan=2, columnspan=8, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		frame_banner_child = frame_banner.nametowidget('child')
		frame_banner_child.grid_columnconfigure(0, weight=1)
		frame_banner_child.grid_rowconfigure(0, weight=1)
		canvas_banner = tk.Canvas(frame_banner_child, bg='black', bd=0, highlightthickness=0, relief='flat')
		canvas_banner.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=(0, 0), pady=(0, 0))
		canvas_banner.after(
			100,
			lambda: [
				canvas_banner.update(),
				# print(canvas_banner.winfo_height()),
				canvas_banner.create_image(canvas_banner.winfo_width(
				)/2, canvas_banner.winfo_height()/2, image=self.photoImg, anchor=CENTER)
			]
		)
		#########
		# SETUP #
		#########
		
		""" self.grid_propagate(False)
		for child in self.winfo_children():
			child.grid_propagate(False)
			for c in child.winfo_children():
				c.grid_propagate(False)
				for _ in c.winfo_children():
					_.grid_propagate(False) """
		
		
		# Set directory
		self.DATA_DIR = Path.cwd() / Path("data")
		self.FILE = ""
		#print("[INJECT] DATA_DIR: {}".format(self.DATA_DIR))
		
		# Load sample file
		load_sample = True
		if(load_sample == True):
			#file = self.DATA_DIR / "sample" / "sample_a2fcf2_lite.csv"
			file = self.DATA_DIR / "sample" / "a0f7db.parquet"
			if(file.is_file()):
				file_load(self.DATA_DIR / file)
			else:
				messagebox.showerror(title="Error", message="[INJECT] Default file not found. No file will be loaded.")
		else:
			self.filecontroller_main.label.child_text.set("No file loaded")