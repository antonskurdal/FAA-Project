import tkinter as tk
from tkinter import *
from pathlib import Path
from tkinter import messagebox
import pandas as pd
from dataclasses import dataclass



from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

import os
import sys
import inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)

parentdir = Path.resolve(Path.cwd()).parent

sys.path.insert(0, parentdir)

print("PARENT DIR: " + str(parentdir))

import util.sku_widgets as sku
import util.grapher as grapher


# Padding
PADX_CONFIG = (2, 2)
PADY_CONFIG = (2, 2)

# Make sure code runs as a module
if(__name__ == '__main__'):
	print("This code is meant to be run as a module.")
	exit(0)

# Methods


# Inject Tab
class Inject(tk.Frame):

	def __init__(self, parent, controller, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs, bg = sku.FRAME_BACKGROUND)
		self.controller = controller

		# Grid Management
		for row in range(7):
			self.grid_rowconfigure(row, weight=1)
		for col in range(18):
			self.grid_columnconfigure(col, weight=1)

		for row in range(7):
			self.grid_rowconfigure(row, weight=0, minsize=100)
		for col in range(18):
			self.grid_columnconfigure(row, weight=0, minsize=100)
		
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



			###########
			# Methods #
			###########

		def populate_listbox(listbox, df, sel=None):
			listbox.delete(0, tk.END)

			# print(df.dtypes)

			# Allow rows with one or less unique value to appear in the selection listboxes
			allow_generic = False


			for col in df.columns:
				if(type(df[col][0]) == str):
					pass
				elif(len(df[col].unique()) <= 1 and allow_generic == False):
					pass
				else:
					listbox.insert(tk.END, col)

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
		'''
		def flash_color(self, color, zone):
			def change_color(color, zone):
				self[zone] = color

			current_color = self[zone]
			next_color = color
			self[zone] = next_color
			self.after(500, lambda: change_color(current_color, zone))
			self.after(1000, lambda: change_color(next_color, zone))
			self.after(1500, lambda: change_color(current_color, zone))
		'''
		def file_browse(var):
			print(var)
			# folder = "data\\test\\"
			# directory = os.path.join(os.getcwd(), folder)
			folder = Path(Path.cwd() / "data" / "test")
			print(folder)

			file = filedialog.askopenfilename(filetypes = [(
				'CSV files', '.csv')], title = "Dataset Selection", initialdir = str(folder))
			if file is None: #askopenfilename return `None` if dialog closed with "cancel".
				return
			var.set(os.path.split(file)[1])

			file_load(file)
			
		def file_load(file):
			base_data = pd.read_csv(file)
			if 'taxonomy' not in base_data.columns:
				base_data.insert(1, 'taxonomy', 'normal')

			populate_listbox(listbox_xs, base_data, 0)
			populate_listbox(listbox_ys, base_data, 2)

			global obj
			obj = CurrentGraph(base_data.copy(deep=True), base_data.copy(deep=True), listbox_xs.get(
				listbox_xs.curselection()), listbox_ys.get(listbox_ys.curselection()))
			
		def reset_plot():
			obj.current = obj.base.copy(deep = True)

		def tag_attacks():

			for i in range(len(obj.base)):

				if((i not in obj.base.index) | (i not in obj.current.index)):
					# print("Skipping " + str(i))
					continue

				if((obj.base.at[i, obj.xs_colname] != obj.current.at[i, obj.xs_colname]) or (obj.base.at[i, obj.ys_colname] != obj.current.at[i, obj.ys_colname])):

					obj.current.at[i, 'taxonomy'] = 'attack'

		def get_fit(master, degree):

			from numpy import array as nparray
			from numpy import polyfit as nppolyfit
			from numpy import poly1d as nppoly1d

			x = nparray(obj.current[obj.xs_colname])
			y = nparray(obj.current[obj.ys_colname])

			map(float, x)
			map(float, y)

			weights = nppolyfit(x, y, degree)
			model = nppoly1d(weights)

			obj.current[obj.xs_colname] = x
			obj.current[obj.ys_colname] = model(x)
			grapher.plotInteractivePolygon(master, obj)

		def file_save(var, data):
			folder = "data\\test\\"
			directory = os.path.join(os.getcwd(), folder)
			f = filedialog.asksaveasfile(filetypes = [('CSV files', '.csv')], mode='w',
										 defaultextension=".csv", initialfile = var.get(), initialdir = str(folder))
			if f is None: #asksaveasfile return `None` if dialog closed with "cancel".
				return

			data.to_csv(f, index = False, line_terminator = '\n')
			f.close()
		
		def sel_changed(event):
			# print(listbox_xs.get(listbox_xs.curselection()))
			# print(listbox_xs.get(listbox_ys.curselection()))

			obj.current = obj.base.copy(deep=True)
			obj.xs_colname = listbox_xs.get(listbox_xs.curselection())
			obj.ys_colname = listbox_ys.get(listbox_ys.curselection())

			self.slider = sku.LiSlider(labelframe_slider, width=500, height=60, min_val=min(obj.current[obj.xs_colname]), max_val=max(
				obj.current[obj.xs_colname]), init_lis=[min(obj.current[obj.xs_colname]), max(obj.current[obj.xs_colname])], show_value=True)
			self.slider.grid(row=0, column=0, rowspan=1, columnspan=3,
							 sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
			self.slider.canv['bg'] = sku.SCALE_BACKGROUND
			self.slider.canv.master['bg'] = sku.SCALE_BACKGROUND
			self.slider.canv['highlightthickness'] = 0
			self.slider.canv.master['highlightthickness'] = 2
			self.slider.canv.master['highlightbackground'] = sku.SCALE_HIGHLIGHTBACKGROUND
		
		def drop_data(master, bounds):
			
			print(bounds)
			
			data = obj.current
			x = obj.xs_colname
			y = obj.ys_colname
			
			low = data[data[x] < bounds[0]]
			
			high = data[data[x] > bounds[1]]
			
			data = pd.concat([low, high])
			
			obj.current = data
			grapher.plotInteractivePolygon(master, obj)	
		
		def add_noise(master, bounds, percent):
			
			from random import randint
			
			print(bounds)
			print(percent)
			
			x = obj.xs_colname
			y = obj.ys_colname
			data = obj.current
			
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
			
			obj.current = data
			grapher.plotInteractivePolygon(master, obj)	
		
		def add_percent(master, bounds, percent):
			x = obj.xs_colname
			y = obj.ys_colname
			data = obj.current
			
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
			
			obj.current = data
			grapher.plotInteractivePolygon(master, obj)		
		
		def add_num(master, bounds, number, do_rolling):
			
			try:
				number = float(number)
			except ValueError:
				messagebox.showerror(title = "Invalid Input", message = "Input for 'Add Numeric' must be a number.")
				return
			
			x = obj.xs_colname
			y = obj.ys_colname
			data = obj.current
			
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
			
			obj.current = data
			grapher.plotInteractivePolygon(master, obj)		
		
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
		
		###########
		# WIDGETS #
		###########

		#############
		# Section 1 #
		#############

		# Main File Loader
		filecontroller_main = sku.FileController(self, text="Load CSV File", label_text="a2fcf2_test_lite.csv", button_text="Browse", button_command=lambda: [file_browse(filecontroller_main.label.child_text), sku.flash_zone(
			filecontroller_main.label, 'bg', 'green'), filecontroller_save_current.label.child_text.set(str(filecontroller_main.label.child_text.get()[:-4]) + "_injected.csv"), sku.flash_zone(filecontroller_save_current.label, 'bg', 'green')])
		filecontroller_main.grid(row=0, column=0, rowspan=1, columnspan=6,
								 sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		# X-Axis Selection
		listbox_xs = sku.CustomListbox(self, selectmode='SINGLE', exportselection=0)
		listbox_xs.grid(row=1, column=0, rowspan=3, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		listbox_xs.bind("<<ListboxSelect>>", sel_changed)

		# Y-Axis Selection
		listbox_ys = sku.CustomListbox(self, selectmode='SINGLE', exportselection=0)
		listbox_ys.grid(row=1, column=3, rowspan=3, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		listbox_ys.bind("<<ListboxSelect>>", sel_changed)

		p = Path.cwd()
		p = Path(p / "data/test")
		print(p)

		# Load defaults
		folder = "data\\test\\"

		directory = p
		#directory = os.path.join(os.getcwd(), folder)

		print(filecontroller_main.label.child_text.get())

		file = Path(directory / filecontroller_main.label.child_text.get())
		print(file)
		if(file.is_file()):
			# if(os.path.isfile(directory + filecontroller_main.label.child_text.get())):
			file_load(directory / filecontroller_main.label.child_text.get())
		else:
			messagebox.showerror(
				title="Error", message="Default file not found.")
			quit()
		
		# Plot
		button_plot = sku.BorderButton(self, button_text='Plot', button_activebackground='green', button_command=lambda: [grapher.plotInteractivePolygon(frame_plot.nametowidget('child'), obj)])
		button_plot.grid(row=4, column=0, rowspan=1, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		# Reset Plot
		button_reset = sku.BorderButton(self, button_text='Reset Plot', button_activebackground='green', button_command=lambda: [reset_plot(), grapher.plotInteractivePolygon(frame_plot.nametowidget('child'), obj)])  # obj.current.set(obj.base),
		button_reset.grid(row=4, column=3, rowspan=1, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		# Print Base Data
		button_print_base = sku.BorderButton(self, button_text='Show Base Data', button_activebackground='green', button_command=lambda: [print(obj.base[[obj.xs_colname, obj.ys_colname, 'taxonomy']])])
		button_print_base.grid(row=5, column=0, rowspan=1, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		# Print Current Data
		button_print_current = sku.BorderButton(self, button_text='Show Current Data', button_activebackground='green', button_command=lambda: [tag_attacks(), print(obj.current[[obj.xs_colname, obj.ys_colname, 'taxonomy']])])
		button_print_current.grid(row=5, column=3, rowspan=1, columnspan=3, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		# Save Modified
		filecontroller_save_current = sku.FileController(self, text="Save Modified CSV File", label_text=filecontroller_main.label.child_text.get()[:-4] + "_modified.csv", button_text="Save", button_command=lambda: [tag_attacks(), file_save(filecontroller_save_current.label.child_text, obj.current), sku.flash_zone(filecontroller_save_current.label, 'bg', 'green')])
		filecontroller_save_current.grid(row=6, column=0, rowspan=1, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		#############
		# Section 2 #
		#############

		# Plot Frame A
		frame_plot = sku.BorderFrame(self, background='#505050', border_color="green")
		frame_plot.grid(row=0, column=6, rowspan=6, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		frame_plot.nametowidget('child').grid_columnconfigure(0, weight=1)  # , minsize = 100)
		frame_plot.nametowidget('child').grid_rowconfigure(0, weight=1)
		frame_plot.nametowidget('child').grid_rowconfigure(1, weight=0, minsize=50)

		# Double Slider Parent Widget
		labelframe_slider = sku.CustomLabelFrame(self, text="X-Axis Modification Range Selection (Inclusive)")
		labelframe_slider.grid(row=6, column=6, rowspan=1, columnspan=6,sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		labelframe_slider.grid_rowconfigure(0, weight=1)
		for col in range(3):
			labelframe_slider.grid_columnconfigure(col, weight=1)
		sel_changed('<<ListboxSelect>>')

		#############
		# Section 3 #
		#############

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

		button_noise = sku.BorderButton(labelframe_noise, button_text='Insert Noise', button_activebackground='green', button_command=lambda: [add_noise(frame_plot.nametowidget('child'), self.slider.getValues(), scale_noise.get())])  # lambda: print(slider.getValues()))
		button_noise.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)

		scale_noise = sku.CustomScale(labelframe_noise, from_=0.0, to=100.0, resolution=1, tickinterval=10, showvalue=True)
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
		self.on = PhotoImage(file=os.path.join(os.getcwd() + "/src/assets/on_und.png"))
		self.off = PhotoImage(file=os.path.join(os.getcwd() + "/src/assets/off_und.png"))
		self.on.zoom(58, 24)
		self.off.zoom(58, 24)

		switch()

		# Banner Image
		img = Image.open(os.getcwd()+"/src/assets/und_banner.png")
		# img = img.resize((516, 125), Image.ANTIALIAS)
		self.photoImg = ImageTk.PhotoImage(img)

		# Banner
		frame_banner = sku.BorderFrame(self, background='#505050', border_color="green")
		frame_banner.grid(row=4, column=12, rowspan=2, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		frame_banner_child = frame_banner.nametowidget('child')
		frame_banner_child.grid_columnconfigure(0, weight=1)
		frame_banner_child.grid_rowconfigure(0, weight=1)

		canvas_banner = tk.Canvas(frame_banner_child, bg='black', bd=0, highlightthickness=0, relief='flat')
		canvas_banner.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=(0, 0), pady=(0, 0))
		canvas_banner.after(
			0,
			lambda: [
				canvas_banner.update(),
				# print(canvas_banner.winfo_height()),
				canvas_banner.create_image(canvas_banner.winfo_width(
				)/2, canvas_banner.winfo_height()/2, image=self.photoImg, anchor=CENTER)
			]
		)

		# WIP Message
		frame_wip = sku.BorderFrame(self, background='#505050', border_color="green")
		frame_wip.grid(row=6, column=12, rowspan=1, columnspan=6, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		frame_wip_child = frame_wip.nametowidget('child')
		frame_wip_child.grid_columnconfigure(0, weight=1)
		frame_wip_child.grid_rowconfigure(0, weight=1)

		label_wip = sku.CustomLabel(frame_wip_child, text="FAA UAS JSON Parser & Attack Injector", font=sku.FONT_BOLD)
		label_wip.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
		label_wip['bg'] = 'black'
		label_wip['font'] = ['Arial', 16, 'bold']


'''

if __name__ == "__main__":
	new = tk.Tk()
	page = Inject(new)
	page.pack(fill="both", expand=True)
	new.mainloop()

'''
