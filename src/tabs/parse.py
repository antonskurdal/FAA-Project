#!/usr/bin/env python

"""This file controls the Parse Tab.

    Parse Tab description.
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from pathlib import Path

import util.sku_widgets as sku
import util.json_parser as json_parser

__author__ = "Anton Skurdal"
__copyright__ = "Copyright 2020, The FAA Project"
__credits__ = ["Anton Skurdal"]
__license__ = "GPL"
__version__ = "1.5"
__maintainer__ = "Anton Skurdal"
__email__ = "antonskurdal@gmail.com"
__status__ = "Development"

#Make sure code runs as a module
if(__name__ == '__main__'):
	print("This code is meant to be run as a module.")
	exit(0)


#Padding
PADX_CONFIG = (2, 2)
PADY_CONFIG = (2, 2)

class Parse(tk.Frame):

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
        
        
        def file_load(file):
            self.FILE = file
            self.filecontroller_main.label.child_text.set(self.FILE.name)
        
        def file_browse(directory, var):
            
            #print("[PARSE][FILE_BROWSE] DIRECTORY: {}".format(directory))
            
            file = Path(filedialog.askopenfilename(
				filetypes = [('JSON files', '.json')],
                title = "JSON File Selection", 
				initialdir = directory))
			
            #print("[PARSE][FILE_BROWSE] FILE NAME: {} (type = {})".format(file.name, type(file)))
            
            #if file is None: #askopenfilename return `None` if dialog closed with "cancel".
            if(file.name == ""): #askopenfilename return `None` if dialog closed with "cancel".
                messagebox.showwarning(title="Warning", message="No file selected")
                return
            else:
                var.set(str(file.name))
                file_load(file)
        
        
        
         #Main File Loader
		#states_2020-05-25-00.json
        self.filecontroller_main = sku.FileController(self, text = "Load CSV File", label_text = "", button_text = "Browse", button_command = lambda:[file_browse(self.DATA_DIR, self.filecontroller_main.label.child_text), sku.flash_zone(self.filecontroller_main.label, 'bg', 'green')])
        self.filecontroller_main.grid(row = 0, column = 0, rowspan = 1, columnspan = 6, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
        
        
        #########
        # SETUP #
        #########
        
        # Set directory
        self.DATA_DIR = Path.cwd() / Path("data")
        self.FILE = ""
        #print("[PARSE] DATA_DIR: {}".format(self.DATA_DIR))
        
        # Load sample file
        load_sample = True
        if(load_sample == True):
            file = self.DATA_DIR / "sample" / "sample_2021-01-01-003250Z.json"
            if(file.exists()):
                file_load(self.DATA_DIR / file)
            else:
                messagebox.showerror(title="Error", message="[PARSE] Default file not found. No file will be loaded.")
            #quit()
        else:
            self.filecontroller_main.label.child_text.set("No file loaded")
        # 	for child in self.winfo_children():
        # 		child['bg'] = 'red'
		
        
        
        
        
        
        
        
        
        
        
        
		
        """  # Load default file
        print("[PARSE] Path.cwd(): {}".format(Path.cwd()))
        directory = Path.cwd() / "data" / "sample"
        default_file = directory / "sample_2021-01-01-003250Z.jsonx"
        
        self.file = ""
        
        #Load defaults
        #folder = "data\\test\\"
        #directory = os.path.join(os.getcwd(), folder)
       
       
        #print(filecontroller_main.label.child_text.get())
        
        
        if(default_file.exists()):
            print("[PARSE] Path Exists")
            self.file = default_file
        else:
            messagebox.showerror(title = "Error", message = "Default file not found.")
            #quit()
        
        if(os.path.isfile(directory + filecontroller_main.label.child_text.get())):
            path = file_load(directory + filecontroller_main.label.child_text.get())
            
        else:
            messagebox.showerror(title = "Error", message = "Default file not found.")
            quit()			 """
		
        
        
       
        
        
        
        
        
        
        
		#Show JSON
        button_show = sku.BorderButton(self, button_text = 'Show JSON', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: [json_parser.show_json(scrolledtext_console, self.FILE)])
        button_show.grid(row = 1, column = 0, rowspan = 1, columnspan = 3, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)		
		
		#Clear Console
        button_clear = sku.BorderButton(self, button_text = 'Clear Console', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: scrolledtext_console.delete(1.0, tk.END))
        button_clear.grid(row = 1, column = 3, rowspan = 1, columnspan = 3, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		
		#Show JSON
        button_keys = sku.BorderButton(self, button_text = 'Show JSON Keys', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: [json_parser.show_keys(scrolledtext_console, self.FILE)])
        button_keys.grid(row = 2, column = 0, rowspan = 1, columnspan = 3, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)		
		
		
		#############
		# Section 2 #
		#############		
		#Console
        frame_console = sku.BorderFrame(self, background = '#505050', border_color = 'green')
        frame_console.grid(row = 0, column = 9, rowspan = 7, columnspan = 9, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
        frame_console.nametowidget('child').grid_rowconfigure(0, weight = 1)
        frame_console.nametowidget('child').grid_columnconfigure(0, weight = 1)
        
        scrolledtext_console = sku.CustomScrolledText(frame_console.nametowidget('child'))
        scrolledtext_console.grid(row = 0, column = 0, rowspan = 1, columnspan = 1, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)

    

'''
if __name__ == "__main__":
    new = tk.Tk()
    page = Parse(new)
    page.pack(fill="both", expand=True)
    new.mainloop()
    '''