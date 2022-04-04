"""
############################
Anton Skurdal's Widget Pack
06/28/2021

Developers: Anton Skurdal


Description:
Custom widget classes for
Tkinter.
############################

"""

###########
# IMPORTS #
###########
import sys
import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as tkscrolled

###################
# MASTER CONTROLS #
###################

#Colors
yellow = "#F5C900"
orange = "#D96000"
red = "#B01C22"
purple = "#9008E7"
offwhite = "#E0E0E0"
lightergray = "#808080"
lightgray = "#606060"
medgray = "#505050"
darkgray = "#404040"

#Font
FONT_NORM = ['Arial', 12]
FONT_BOLD = ['Arial', 12, 'bold']
FONT_SMALL = ['Arial', 10]
if (sys.platform == "linux"):
	FONT_NORM = ['Ubuntu', 12]
	FONT_BOLD = ['Ubuntu', 12, 'bold']
	FONT_SMALL = ['Ubuntu', 10]
FONT_COLOR = offwhite
WIDGET_RELIEF = 'flat'

#App
APP_BACKGROUND = "blue"
#Frame
FRAME_BACKGROUND = lightergray
#Button
BUTTON_BACKGROUND = darkgray
BUTTON_ACTIVEBACKGROUND = 'green'
BUTTON_FOREGROUND = offwhite
BUTTON_ACTIVEFOREGROUND = offwhite
BUTTON_FONT = FONT_NORM
BUTTON_BORDERWIDTH = 0
BUTTON_RELIEF = 'flat'
BUTTON_RELIEF_PRESSED = 'groove'
BUTTON_CURSOR = "hand2"
#Label
#Entry
ENTRY_FONT = FONT_NORM
ENTRY_BACKGROUND = lightgray
ENTRY_BORDERCOLOR = orange
ENTRY_HIGHLIGHTCOLOR = "#009A44"
ENTRY_HIGHLIGHTBACKGROUND = "#404040"
#LabelFrame
LABELFRAME_FONT = FONT_BOLD
#ListBox
LISTBOX_BACKGROUND = '#606060'
LISTBOX_BORDERCOLOR = '#404040'
LISTBOX_RELIEF = WIDGET_RELIEF


#ScrolledText
SCROLLEDTEXT_RELIEF = WIDGET_RELIEF

#Scale
SCALE_BACKGROUND = medgray
SCALE_TROUGHCOLOR = lightgray
SCALE_FONT = FONT_SMALL
SCALE_FOREGROUND = FONT_COLOR
SCALE_HIGHLIGHTBACKGROUND = darkgray
SCALE_ACTIVEBACKGROUND = orange

#Padding
PADX_CONFIG = (2, 2)
PADY_CONFIG = (2, 2)


#Make sure code runs as a module
if(__name__ == '__main__'):
	print("This code is meant to be run as a module.")
	exit(0)

def get_attributes(widget):
	#Method is for debugging widget args
	widg = widget
	keys = widg.keys()
	for key in keys:
		print("Attribute: {:<20}".format(key), end=' ')
		value = widg[key]
		vtype = type(value)
		print('Type: {:<30} Value: {}'.format(str(vtype), value))

def iter_layout(layout, tab_amnt=0, elements=[]):
	#CODE SOURCE: https://stackoverflow.com/questions/45389166/how-to-know-all-style-options-of-a-ttk-widget/48933106
	"""Recursively prints the layout children."""
	el_tabs = '  '*tab_amnt
	val_tabs = '  '*(tab_amnt + 1)

	for element, child in layout:
		elements.append(element)
		print(el_tabs+ '\'{}\': {}'.format(element, '{'))
		for key, value in child.items():
			if type(value) == str:
				print(val_tabs + '\'{}\' : \'{}\','.format(key, value))
			else:
				print(val_tabs + '\'{}\' : [('.format(key))
				iter_layout(value, tab_amnt=tab_amnt+3)
				print(val_tabs + ')]')

		print(el_tabs + '{}{}'.format('} // ', element))

	return elements

def stylename_elements_options(stylename, widget):
	#CODE SOURCE: https://stackoverflow.com/questions/45389166/how-to-know-all-style-options-of-a-ttk-widget/48933106
	"""Function to expose the options of every element associated to a widget stylename."""

	try:
		# Get widget elements
		style = ttk.Style()
		layout = style.layout(stylename)
		config = widget.configure()

		print('{:*^50}\n'.format(f'Style = {stylename}'))

		print('{:*^50}'.format('Config'))
		for key, value in config.items():
			print('{:<15}{:^10}{}'.format(key, '=>', value))

		print('\n{:*^50}'.format('Layout'))
		elements = iter_layout(layout)

		# Get options of widget elements
		print('\n{:*^50}'.format('element options'))
		for element in elements:
			print('{0:30} options: {1}'.format(
		    element, style.element_options(element)))
	
	#Example usage:
	#widget = ttk.Button(None)
	#class_ = widget.winfo_class()
	#stylename_elements_options(class_, widget)	
	
	except tk.TclError:
		print('_tkinter.TclError: "{0}" in function'
	      'widget_elements_options({0}) is not a regonised stylename.'
		.format(stylename))

def flash_zone(self, zone, data):
			current_data = self[zone]
			next_data = data
			self[zone] = next_data
			
			self.after(500, lambda: change(zone, current_data))
			self.after(1000, lambda: change(zone, next_data))
			self.after(1500, lambda: change(zone, current_data))
			
			#This exists because values can't be assigned in expressions
			def change(zone, data):
				self[zone] = data

def resize(self, xstart, xstop, xstep, ystart, ystop, ystep):
	
	#Check if desired size has been reached
	if(self.winfo_width() == xstop and self.winfo_height() == ystop):
		return
	
	#Check if steps are infinite
	if(xstep == 0 and ystep == 0):
		return	
	
	#Check if steps need to be negative
	if(xstop < xstart and xstep > 0):
		xstep = xstep * -1
	if(ystop < ystart and ystep > 0):
		ystep = ystep * -1
	
	#Check if next step is final step
	if(xstep > 0 and xstart+xstep >= xstop):
		self.geometry(str(xstop)+"x"+str(self.winfo_height()))
		self.update_idletasks()
		xstep = 0
		xstart = xstop
	elif(xstep < 0 and xstart+xstep <= xstop):
		self.geometry(str(xstop)+"x"+str(self.winfo_height()))
		self.update_idletasks()
		xstep = 0
		xstart = xstop
	elif(ystep > 0 and ystart+ystep >= ystop):
		self.geometry(str(self.winfo_width())+"x"+str(ystop))
		self.update_idletasks()
		ystep = 0
		ystart = ystop
	elif(ystep < 0 and ystart+ystep <= ystop):
		self.geometry(str(self.winfo_width())+"x"+str(ystop))
		self.update_idletasks()
		ystep = 0
		ystart = ystop
	
	#Update geometry
	self.geometry(str(xstart)+"x"+str(ystart))
	self.update_idletasks()
	
	#Recursive call
	self.after(0, resize(self, xstart+xstep, xstop, xstep, ystart+ystep, ystop, ystep))

##################
# CUSTOM WIDGETS #
##################
class CustomButton(tk.Button):
	def __init__(self, master, **kw):
		tk.Button.__init__(self, master=master, **kw)

		#if (self['font'] == "TkDefaultFont"):
			#self['font'] = FONT_NORM	
		#else:
			#self['font'] = self['font']


		self['font'] = BUTTON_FONT

		if (self['activebackground'] == "SystemButtonFace"):
			self['activebackground'] = "red"


		self.bg = BUTTON_BACKGROUND

		if (self.master.winfo_class() == "Labelframe" or "Frame"):
			self.master['bg'] = self.bg

		#elif (self.master.winfo_class() == "Frame"):
		#	self.master['bg'] = self.bg		


		self['background'] = self.bg
		self['foreground'] = BUTTON_FOREGROUND
		self['activeforeground'] = BUTTON_FOREGROUND
		self['borderwidth'] = BUTTON_BORDERWIDTH
		self['relief'] = BUTTON_RELIEF
		
		self.bind("<Enter>", self.on_enter)
		self.bind("<Leave>", self.on_leave)
		self.bind("<Button-1>", self.on_click)
		self.bind("<ButtonRelease-1>", self.on_release)

	def on_enter(self, e):
		if(self['state'] == 'disabled'):
			return		
		
		self['cursor'] = BUTTON_CURSOR
		
		if (self.master.winfo_class() == "Labelframe" or "Frame"):
			self.master['bg'] = self['activebackground']


	def on_leave(self, e):
		if(self['state'] == 'disabled'):
			return			
		
		self['cursor']="arrow"
		self['background'] = self.bg
		
		if (self.master.winfo_class() == "Labelframe" or "Frame"):
			self.master['bg'] = self['bg']

	def on_click(self, e):
		if(self['state'] == 'disabled'):
			return			
		
		self['background'] = BUTTON_ACTIVEBACKGROUND
		self['relief'] = BUTTON_RELIEF_PRESSED

	def on_release(self, e):
		if(self['state'] == 'disabled'):
			return			
		
		if (self.master.winfo_class() == "Labelframe" or "Frame"):
			self.master['bg'] = self['activebackground']	

		self['background'] = self.bg
		self['relief'] = BUTTON_RELIEF

class CustomLabel(tk.Label):
	def __init__(self, master, **kw):
		tk.Button.__init__(self, master=master, **kw)

		if (self['font'] == "TkDefaultFont"):
			self['font'] = FONT_NORM		

		self['state'] = 'disabled'

		#self['background'] = "#626262"
		self['background'] = "#404040"
		self['disabledforeground'] = FONT_COLOR
		self['relief'] = 'flat'

class CustomEntry(tk.Entry):
	def __init__(self, master, **kw):
		tk.Entry.__init__(self, master=master, **kw)

		self['highlightthickness'] = 2
		self['highlightbackground'] = ENTRY_HIGHLIGHTBACKGROUND
		#self['highlightcolor'] = "#D96000"
		self['highlightcolor'] = ENTRY_HIGHLIGHTCOLOR
		self['selectbackground'] = ENTRY_HIGHLIGHTCOLOR

		self['disabledbackground'] = ENTRY_BACKGROUND
		self['disabledforeground'] = FONT_COLOR
		self['readonlybackground'] = ENTRY_BACKGROUND


		self['bg'] = "#606060"
		self['fg'] = FONT_COLOR
		self['font'] = ENTRY_FONT
		#self['state'] = NORMAL
		self['relief'] = 'flat'

class CustomLabelFrame(tk.LabelFrame):
	def __init__(self, master, **kw):
		tk.LabelFrame.__init__(self, master=master, **kw)

		#self['highlightthickness'] = 2
		self['highlightbackground'] = "#404040"
		self['highlightcolor'] = "#404040"


		self['bg'] = "#505050"
		self['fg'] = FONT_COLOR
		self['font'] = LABELFRAME_FONT
		#self['state'] = NORMAL
		self['relief'] = 'flat'

class CustomListbox(tk.Listbox):
	def __init__(self, master, **kw):
		tk.Listbox.__init__(self, master, **kw)

		#get_attributes(self)
		#print(self.config())
		
		self['activestyle'] = 'none'
		self['bg'] = LISTBOX_BACKGROUND
		self['fg'] = FONT_COLOR
		self['font'] = FONT_NORM
		self['state'] = 'normal'
		#self['disabledforeground'] = 'red'
		self['highlightcolor'] = LISTBOX_BORDERCOLOR
		self['highlightbackground'] = LISTBOX_BORDERCOLOR
		self['highlightthickness'] = 0
		self['selectbackground'] = ENTRY_HIGHLIGHTCOLOR
		self['relief'] = LISTBOX_RELIEF
		self['selectborderwidth'] = 0
		self['highlightthickness'] = 2
		#get_attributes(self)
		
		self.scrollbar = tk.Scrollbar(self, troughcolor = 'red')
		self.scrollbar.pack(side = 'right', fill = 'both')
		self.config(yscrollcommand = self.scrollbar.set)
		self.scrollbar.config(command = self.yview)
		
		self.bind("<<ListboxSelect>>", self.on_select)
	
	def on_select(self, e):
		self['highlightbackground'] = LISTBOX_BORDERCOLOR

class CustomScrolledText(tkscrolled.ScrolledText):
	def __init__(self, master, **kw):
		tkscrolled.ScrolledText.__init__(self, master, **kw)

		#get_attributes(self)
		#print(self.config())

		self['bg'] = master['bg']
		self['fg'] = FONT_COLOR
		self['font'] = FONT_NORM
		self['highlightcolor'] = ENTRY_HIGHLIGHTCOLOR
		self['highlightbackground'] = ENTRY_HIGHLIGHTCOLOR
		self['selectbackground'] = ENTRY_HIGHLIGHTCOLOR
		self['relief'] = SCROLLEDTEXT_RELIEF

class BorderButton(tk.Frame):
	def __init__(self, master, button_text = None, button_activebackground = None, button_command = None, button_image = '', **kw):
		tk.Frame.__init__(self, master=master, **kw)

		self.grid_rowconfigure(0, weight = 1)
		self.grid_columnconfigure(0, weight = 1)
		
	
		
		#test = DynamicBorderButton(self, button_text = "Sup", button_activebackground = orange)
		#test.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)		

		self.child = CustomButton(self, text = button_text, activebackground = button_activebackground, command = button_command, image = button_image)
		self.child.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)

class CustomScale(tk.Scale):
	def __init__(self, master, **kw):
		tk.Scale.__init__(self, master=master, **kw)

		#print(self.config())

		self['background'] = SCALE_BACKGROUND
		self['foreground'] = SCALE_FOREGROUND
		self['troughcolor'] = SCALE_TROUGHCOLOR
		self['font'] = SCALE_FONT
		self['highlightbackground'] = SCALE_HIGHLIGHTBACKGROUND
		self['bd'] = 0
		self['orient'] = 'horizontal'

		self['sliderrelief'] = 'solid'
		self['sliderlength'] = 50

		self['highlightthickness'] = 2
		self['highlightcolor'] = 'red'
		self['activebackground'] = SCALE_ACTIVEBACKGROUND
		self['relief'] = 'flat'

class BorderFrame(tk.Frame):
	def __init__(self, master, border_color, **kw):
		tk.Frame.__init__(self, master=master, **kw)

		self.grid_rowconfigure(0, weight = 1)
		self.grid_columnconfigure(0, weight = 1)
		
		child = tk.Frame(self, name = 'child', background = self['bg'])
		child.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		child.grid_propagate(False)
		
		self['bg'] = border_color

class BorderLabel(tk.Frame):
	def __init__(self, master, **kw):
		tk.Frame.__init__(self, master=master, **kw)
		
		self.child_text = tk.StringVar()
		
		self['bg'] = ENTRY_HIGHLIGHTBACKGROUND
		self.grid_rowconfigure(0, weight = 1)
		self.grid_columnconfigure(0, weight = 1)
		
		self.child = tk.Label(self, name = 'child', textvariable = self.child_text)
		self.child.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		#self.child.grid_propagate(False)
		
		self.child['font'] = ENTRY_FONT
		self.child['foreground'] = FONT_COLOR
		self.child['background'] = ENTRY_BACKGROUND
		self.child['relief'] = WIDGET_RELIEF
		self.child['activebackground'] = 'red'	

class FileController(CustomLabelFrame):
	def __init__(self, master, label_text = None, button_text = None, button_command = None, **kw):
		CustomLabelFrame.__init__(self, master=master, **kw)
		
		self.grid_rowconfigure(0, weight = 1)
		self.grid_columnconfigure(0, weight = 1)
		self.grid_columnconfigure(1, weight = 1)
		self.grid_columnconfigure(2, weight = 1)
		
		self.label_text = label_text
		self.button_text = button_text
		self.button_command = button_command
		
		
		self.label = BorderLabel(self)
		self.label.grid(row = 0, column = 0, rowspan = 1, columnspan = 2, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		self.label.child_text.set(label_text)
		self.label.grid_propagate(False)
		
		self.button = BorderButton(self, button_text = self.button_text, button_activebackground = BUTTON_ACTIVEBACKGROUND, button_command = button_command)
		self.button.grid(row = 0, column = 2, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		self.button.grid_propagate(False)

class LiSlider(tk.Frame):
	#AUTHOR: MengxunLi
	#LINK: https://github.com/MenxLi/tkSliderWidget/blob/master/tkSliderWidget.py
	#BSD 2-Clause License
	#Copyright (c) 2020, Mengxun Li
	#All rights reserved.
	
	LINE_COLOR = "#009A44"
	LABEL_COLOR = 'white'
	LINE_WIDTH = 3
	BAR_COLOR_INNER = "#009A44"
	BAR_COLOR_OUTTER = "white"
	BAR_RADIUS = 10
	BAR_RADIUS_INNER = BAR_RADIUS-4
	DIGIT_PRECISION = '.1f' # for showing in the canvas
	def __init__(self, master, width = 400, height = 80, min_val = 0, max_val = 1, init_lis = None, show_value = True):
		tk.Frame.__init__(self, master, height = height, width = width)
		self.master = master
		if init_lis == None:
			init_lis = [min_val]
		self.init_lis = init_lis
		self.max_val = max_val
		self.min_val = min_val
		self.show_value = show_value
		self.H = height
		self.W = width
		self.canv_H = self.H
		self.canv_W = self.W
		if not show_value:
			self.slider_y = self.canv_H/2 # y pos of the slider
		else:
			self.slider_y = self.canv_H*2/5
		self.slider_x = LiSlider.BAR_RADIUS + 40 # x pos of the slider (left side)

		self.bars = []
		self.selected_idx = None # current selection bar index
		for value in self.init_lis:
			pos = (value-min_val)/(max_val-min_val)
			ids = []
			bar = {"Pos":pos, "Ids":ids, "Value":value}
			self.bars.append(bar)


		self.canv = tk.Canvas(self, height = self.canv_H, width = self.canv_W)
		self.canv.pack()
		self.canv.bind("<Motion>", self._mouseMotion)
		self.canv.bind("<B1-Motion>", self._moveBar)

		self.__addTrack(self.slider_x, self.slider_y, self.canv_W-self.slider_x, self.slider_y)#self.canv_W-self.slider_x, self.slider_y)
		for bar in self.bars:
			bar["Ids"] = self.__addBar(bar["Pos"])


	def getValues(self):
		values = [bar["Value"] for bar in self.bars]
		return sorted(values)

	def _mouseMotion(self, event):
		x = event.x; y = event.y
		selection = self.__checkSelection(x,y)
		if selection[0]:
			self.canv.config(cursor = "hand2")
			self.selected_idx = selection[1]
		else:
			self.canv.config(cursor = "")
			self.selected_idx = None

	def _moveBar(self, event):
		x = event.x; y = event.y
		if self.selected_idx == None:
			return False
		pos = self.__calcPos(x)
		idx = self.selected_idx
		self.__moveBar(idx,pos)

	def __addTrack(self, startx, starty, endx, endy):
		id1 = self.canv.create_line(startx, starty, endx, endy, fill = LiSlider.LINE_COLOR, width = LiSlider.LINE_WIDTH)
		return id

	def __addBar(self, pos):
		"""@ pos: position of the bar, ranged from (0,1)"""
		if pos <0 or pos >1:
			raise Exception("Pos error - Pos: "+str(pos))
		R = LiSlider.BAR_RADIUS
		r = LiSlider.BAR_RADIUS_INNER
		L = self.canv_W - 2*self.slider_x
		y = self.slider_y
		x = self.slider_x+pos*L
		id_outer = self.canv.create_oval(x-R,y-R,x+R,y+R, fill = LiSlider.BAR_COLOR_OUTTER, width = 2, outline = "")
		id_inner = self.canv.create_oval(x-r,y-r,x+r,y+r, fill = LiSlider.BAR_COLOR_INNER, outline = "")
		if self.show_value:
			y_value = y+LiSlider.BAR_RADIUS+8
			value = pos*(self.max_val - self.min_val)+self.min_val
			id_value = self.canv.create_text(x,y_value, fill = LiSlider.LABEL_COLOR, text = format(value, LiSlider.DIGIT_PRECISION))
			return [id_outer, id_inner, id_value]
		else:
			return [id_outer, id_inner]

	def __moveBar(self, idx, pos):
		ids = self.bars[idx]["Ids"]
		for id in ids:
			self.canv.delete(id)
		self.bars[idx]["Ids"] = self.__addBar(pos)
		self.bars[idx]["Pos"] = pos
		self.bars[idx]["Value"] = pos*(self.max_val - self.min_val)+self.min_val

	def __calcPos(self, x):
		"""calculate position from x coordinate"""
		pos = (x - self.slider_x)/(self.canv_W-2*self.slider_x)
		if pos<0:
			return 0
		elif pos>1:
			return 1
		else:
			return pos

	def __getValue(self, idx):
		"""#######Not used function#####"""
		bar = self.bars[idx]
		ids = bar["Ids"]
		x = self.canv.coords(ids[0])[0] + LiSlider.BAR_RADIUS
		pos = self.__calcPos(x)
		return pos*(self.max_val - self.min_val)+self.min_val

	def __checkSelection(self, x, y):
		"""
		To check if the position is inside the bounding rectangle of a Bar
		Return [True, bar_index] or [False, None]
		"""
		for idx in range(len(self.bars)):
			id = self.bars[idx]["Ids"][0]
			bbox = self.canv.bbox(id)
			if bbox[0] < x and bbox[2] > x and bbox[1] < y and bbox[3] > y:
				return [True, idx]
		return [False, None]

class CustomSwitch(tk.Frame):
	"""
	EXAMPLE USAGE:
	
	# TEST SWITCH
	on_image = PhotoImage(file=Path.cwd() / "src" / "assets" / "on_und.png")
	off_image = PhotoImage(file=Path.cwd() / "src" / "assets" / "off_und.png")
	test = sku.CustomSwitch(self, text="Test", textanchor = "n", on_image = on_image, off_image=off_image, init_state = False)
	test.grid(row=4, column=12, rowspan=1, columnspan=2, sticky="NSEW", padx=PADX_CONFIG, pady=PADY_CONFIG)
	print("[INJECT] state: {}".format(test.get_state()))
	"""
	def __init__(self, master, text, textanchor, on_image, off_image, init_state, **kw):
		tk.Frame.__init__(self, master=master, **kw)
		
		self.grid_rowconfigure(0, weight=1)
		self.grid_columnconfigure(0, weight=1)
		
		self.text = text
		self.textanchor = textanchor
		self.on_image = on_image
		self.off_image = off_image
		self.state = init_state
		
		def toggle():
			
			if self.state == True:
				_ = list(self.labelframe.pack_slaves())
				for i in _:
					i.pack_forget()
				
				self.switch = BorderButton(self.labelframe, button_image = self.off_image, button_activebackground = '#404040', button_command=toggle)
				self.switch.pack()
				self.state = False
				#print(self.state)
				
			else:
				_ = list(self.labelframe.pack_slaves())
				for i in _:
					i.pack_forget()
				
				self.switch = BorderButton(self.labelframe, button_image = self.on_image, button_activebackground = '#404040', button_command=toggle)
				self.switch.pack()
				self.state = True
				#print(self.state)
		
		
		
		# Create labelframe container
		self.labelframe = CustomLabelFrame(self, text = self.text, labelanchor = textanchor)
		self['bg'] = self.master['bg']
		self.labelframe['font'] = FONT_NORM
		self.labelframe.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, padx = PADX_CONFIG, pady = PADY_CONFIG)
		
		# Create switch by toggling
		toggle()
		
		# Toggle again if switch is not at correct initial state
		if(self.state != init_state):
			toggle()
		
	def get_state(self):
		return self.state


# Custom Radiobutton
RADIOBUTTON_ACTIVEBACKGROUND = "#009A44"
RADIOBUTTON_ACTIVEFOREGROUND = FONT_COLOR

RADIOBUTTON_BACKGROUND = "#404040"
RADIOBUTTON_FOREGROUND = FONT_COLOR
RADIOBUTTON_SELECTCOLOR = "#009A44"
RADIOBUTTON_ACTIVEBORDERCOLOR = "white"

RADIOBUTTON_FONT = FONT_NORM

RADIOBUTTON_RELIEF = "flat"
RADIOBUTTON_OVERRELIEF = "flat"
RADIOBUTTON_OFFRELIEF = "flat"
RADIOBUTTON_RELIEF_PRESSED = "groove"
RADIOBUTTON_CURSOR = BUTTON_CURSOR
class BorderRadiobutton(tk.Frame):
	def __init__(self, master, activebordercolor, text, variable, command, value, indicator, **kw):
		tk.Frame.__init__(self, master=master, **kw)
		
		self.activebordercolor = activebordercolor
		self.text = text
		self.variable = variable
		self.command = command
		self.value = value
		self.indicator = indicator
		
		
		self['background'] = RADIOBUTTON_BACKGROUND
		self.grid_rowconfigure(0, weight = 1)
		self.grid_columnconfigure(0, weight = 1)
		
		self.radiobutton = tk.Radiobutton(self, text = self.text, variable = self.variable, command = self.command, value = self.value, indicator = self.indicator)
		self.radiobutton.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		
		
		
		
		self.radiobutton['activebackground'] = RADIOBUTTON_ACTIVEBACKGROUND
		self.radiobutton['activeforeground'] = RADIOBUTTON_ACTIVEFOREGROUND
		
		self.radiobutton['background'] = RADIOBUTTON_BACKGROUND
		self.radiobutton['foreground'] = RADIOBUTTON_FOREGROUND
		self.radiobutton['selectcolor'] = RADIOBUTTON_SELECTCOLOR
		
		self.radiobutton['font'] = RADIOBUTTON_FONT
		
		self.radiobutton['relief'] = RADIOBUTTON_RELIEF
		self.radiobutton['overrelief'] = RADIOBUTTON_OVERRELIEF
		self.radiobutton['offrelief'] = RADIOBUTTON_OFFRELIEF
		
		#self.radiobutton['highlightcolor'] = 'pink'
		self.radiobutton['highlightthickness'] = 0
		#self.radiobutton['highlightbackground'] = 'red'
		
		self.radiobutton['borderwidth'] = 0
		
		#self.radiobutton['disabledforeground'] = 'red'
		
		
		
		
		
		# self.bind("<Enter>", self.on_enter)
		# self.bind("<Leave>", self.on_leave)
		# self.bind("<Button-1>", self.on_click)
		# self.bind("<ButtonRelease-1>", self.on_release)
		
		self.radiobutton.bind("<Enter>", self.on_enter)
		self.radiobutton.bind("<Leave>", self.on_leave)
		self.radiobutton.bind("<Button-1>", self.on_click)
		self.radiobutton.bind("<ButtonRelease-1>", self.on_release)
	
	""" def invoke(self):
		self.radiobutton.invoke()
		self['background'] = self.activebordercolor
		#self['background'] = RADIOBUTTON_ACTIVEBORDERCOLOR
		self.update_idletasks() """
		
	
		
	def on_enter(self, e):
		if(self.radiobutton['state'] == 'disabled'):
			return
		
		self['cursor'] = RADIOBUTTON_CURSOR
		
		# if(self.variable.get() == self.value):
		# 	return
		# else:
		# 	self['background'] = self.activebordercolor
		


	def on_leave(self, e):
		if(self.radiobutton['state'] == 'disabled'):
			return			
		
		self['cursor']="arrow"
		
		# if(self.variable.get() == self.value):
		# 	return
		# else:
		# 	self['background'] = RADIOBUTTON_BACKGROUND
	
	
	def on_click(self, e):
		if(self.radiobutton['state'] == 'disabled'):
			return			
		
		self['relief'] = BUTTON_RELIEF_PRESSED
		
		#self['background'] = self.activebordercolor
		
		#self['background'] = "pink"
		
		#self['background'] = "pink"
		
		# if(self.variable.get() == self.value):
		# 	return
		# else:
		# 	self['background'] = RADIOBUTTON_BACKGROUND
		
		
		
		#self['background'] = BUTTON_ACTIVEBACKGROUND
		

	def on_release(self, e):
		if(self.radiobutton['state'] == 'disabled'):
			return
		
		self['relief'] = BUTTON_RELIEF
		
		#self['background'] = "white"
		
		
		#self['background'] = "pink"
		
		# if(self.variable.get() == self.value):
		# 	return
		# else:
		# 	self['background'] = "pink"
		
		#self.update_idletasks()
		
		# else:
		# 	self['background'] = RADIOBUTTON_BACKGROUND
		
	


""" class BorderCheckbutton(tk.Frame):
	def __init__(self, master, activebordercolor, text, variable, command, value, indicator, **kw):
		tk.Frame.__init__(self, master=master, **kw) """
		
		
		
		