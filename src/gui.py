#!/usr/bin/env python

"""This file controls the root window on which the rest of the application is built.
	
	It creates a root window on which widgets can be added. The main widgets are tabs
	for this application. Tabs are found in src/tabs and are classes. Each tab has its
	own unique GUI. They are shown when the user presses the button for that tab,
	which must be added here in the root. 
"""

import tkinter as tk

from tabs.inject import Inject
from tabs.parse import Parse
from tabs.live import LiveData
import util.sku_widgets as sku

__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2020, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"



#sku.ENTRY_HIGHLIGHTCOLOR = "#009A44"
PADX_CONFIG = (2,2)
PADY_CONFIG = (2,2)


class App(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.title('UND FAA-A44 Research Project - Dataset Injector/Modifier & Aircraft Live Tracker')
		self.geometry("1800x750")
		self.resizable(0, 0)
		self['background'] = sku.APP_BACKGROUND
		master = self

		for row in range(2):
			self.grid_rowconfigure(row, weight = 1)		
		for col in range(1):
			self.grid_columnconfigure(col, weight = 1)
		
		self.grid_rowconfigure(0, weight = 0, minsize = 50)
		#self.grid_columnconfigure(0, weight = 1)


		self.tab_frame = tk.Frame(master, bg = sku.FRAME_BACKGROUND)
		self.tab_frame.grid(row = 0, column = 0, sticky = "NSEW")
		self.tab_frame.grid_rowconfigure(0, weight = 1)
		self.tab_frame.grid_columnconfigure(0, weight = 1)
		self.tab_frame.grid_columnconfigure(1, weight = 1)
		self.tab_frame.grid_columnconfigure(2, weight = 1)	
		
		#Analyze
		self.button_Inject = sku.BorderButton(self.tab_frame, button_text="Inject", button_command=lambda: self.show_frame("Inject"), button_activebackground = sku.BUTTON_ACTIVEBACKGROUND)
		self.button_Inject.grid(row = 0, column = 0, sticky="NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		
		#Settings
		self.button_Parse = sku.BorderButton(self.tab_frame, button_text="Parse JSON", button_command=lambda: self.show_frame("Parse"), button_activebackground = sku.BUTTON_ACTIVEBACKGROUND)
		self.button_Parse.grid(row = 0, column = 1, sticky="NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)

		#Live API
		self.button_live = sku.BorderButton(self.tab_frame, button_text="Live Data", button_command=lambda: self.show_frame("LiveData"), button_activebackground = sku.BUTTON_ACTIVEBACKGROUND)
		self.button_live.grid(row = 0, column = 2, sticky="NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		
		
		#Main Container for frame stacking
		#self.container = tk.Frame(master, bg='red')
		self.container = tk.Frame(master, bg='white')
		self.container.grid(row = 1, column = 0, sticky = "NSEW")
		self.container.grid_columnconfigure(0, weight = 1)
		self.container.grid_rowconfigure(0, weight = 1)
		
		self.frames = {}
		for Tab in (Inject, Parse, LiveData):
			page_name = Tab.__name__
			frame = Tab(parent = self.container, controller = self)
			self.frames[page_name] = frame
			
			#Put all of the pages in the same place
			#The frame on top will be visible
			frame.grid(row = 0, column = 0, sticky = "NSEW")
		
		self.show_frame("Inject")
		
	def show_frame(self, page_name):
		#Show the frames for the given page name
		frame = self.frames[page_name]
		frame.tkraise()

if __name__ == '__main__':
	app = App()
	app.mainloop()	