#Make sure code runs as a module
if(__name__ == '__main__'):
	print("This code is meant to be run as a module.")
	exit(0)

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import util.sku_widgets as sku
import util.json_parser as json_parser


import os


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
            folder = "data\\test\\"
            directory = os.path.join(os.getcwd(), folder)
            file = filedialog.askopenfilename(filetypes = [('JSON files', '.json')], title = "JSON File Selection", initialdir = str(folder))
            if file is None: #askopenfilename return `None` if dialog closed with "cancel".
                return
            var.set(os.path.split(file)[1])
        
        def file_load(file):
            path = file
            return path
        
        #############
		# Section 1 #
		#############
		#Main File Loader
		#states_2020-05-25-00.json
        filecontroller_main = sku.FileController(self, text = "Load CSV File", label_text = "2021-01-01-003250Z.json", button_text = "Browse", button_command = lambda:[file_browse(filecontroller_main.label.child_text), sku.flash_zone(filecontroller_main.label, 'bg', 'green')])
        filecontroller_main.grid(row = 0, column = 0, rowspan = 1, columnspan = 6, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		
		#Load defaults
        folder = "data\\test\\"
        directory = os.path.join(os.getcwd(), folder)
        #print(filecontroller_main.label.child_text.get())
        if(os.path.isfile(directory + filecontroller_main.label.child_text.get())):
            path = file_load(directory + filecontroller_main.label.child_text.get())
        else:
            messagebox.showerror(title = "Error", message = "Default file not found.")
            quit()			
		
		#Show JSON
        button_show = sku.BorderButton(self, button_text = 'Show JSON', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: [json_parser.show_json(scrolledtext_console, path)])
        button_show.grid(row = 1, column = 0, rowspan = 1, columnspan = 3, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)		
		
		#Clear Console
        button_clear = sku.BorderButton(self, button_text = 'Clear Console', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: scrolledtext_console.delete(1.0, tk.END))
        button_clear.grid(row = 1, column = 3, rowspan = 1, columnspan = 3, sticky = "NSEW", padx = PADX_CONFIG, pady = PADY_CONFIG)
		
		#Show JSON
        button_keys = sku.BorderButton(self, button_text = 'Show JSON Keys', button_activebackground = sku.BUTTON_ACTIVEBACKGROUND, button_command = lambda: [json_parser.show_keys(scrolledtext_console, path)])
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