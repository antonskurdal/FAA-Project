import tkinter as tk
from tkinter import Tk, ttk

from PIL import Image
from PIL import ImageTk
from pathlib import Path

# audio module works as expected
#import audio_module as am

# I want this window to open and close on command
from tabs.inject import Inject
from tabs.parse import Parse
from tabs.live import LiveData

import util.sku_widgets as sku

'''
root = tk.Tk()
root.title("Test")
root.geometry("400x400")
'''

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
        
        #style = ttk.Style()

        #style.theme_create('skurdal', parent = 'clam')
        #print(style.theme_names())
        #style.theme_use('skurdal')

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
        self.container = tk.Frame(master, bg='red')
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

        '''
    #def __init__(self, master):
        self.notebook = ttk.Notebook(master, )
        self.notebook.grid()

        self.myButton = tk.Button(master, text = "Click Me!", command = self.clicker)
        self.myButton.grid()
        self.myFrame1 = tk.Frame(master, bg = "blue", width = 300, height = 300)
        self.myFrame1.grid()

        self.myFrame2 = tk.Frame(master, bg = "red")
        self.myFrame2.grid()


        

        self.notebook.add(self.myFrame1, text = "BLUE TAB")
        self.notebook.add(self.myFrame2, text = "RED TAB")


    def clicker(self):
        print("HELLO YOU CLICKED A BUTTON")
        Page(self.myFrame1)
        '''
    def show_frame(self, page_name):
        #Show the frames for the given page name
        frame = self.frames[page_name]
        frame.tkraise()



"""
class GUI(tk.Frame):

    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        #self.geometry("800x800")

        self.notebook = ttk.Notebook(GUI)

        #self.new = tk.Toplevel(self) # auto loads a second, unwanted window

        self.session_counter = 0
        self.start_btn = tk.Button(root, text="start", command=self.start)
        self.start_btn.pack()
        #self.start_btn.grid(row=4,column=0,sticky="nsew",pady=30, padx=30, ipady=18)


    def start(self):
        #am.spell() # these audio imports work like a charm, every btn press - single functions call OK

        self.session_counter += 1
        print(self.session_counter)

        #import simple_module - if used here, my usual 'illegal' import style (works great, once only,
        # unless in same script as __main__ in which case all re-imports work fine)

        # Import attempts
        #import simple_module as sm
        #page = Page(new) # Page not defined
        #sm.Page() #missing parent arg (new)

        # error: 'new' not defined
        #sm.Page(new)


if __name__ == '__main__':
    print('running as __main__')
root = tk.Tk()
page = Page(root)
page.pack(fill="both", expand=True)
main = GUI(root)

root.mainloop()
"""
if __name__ == '__main__':
	app = App()
	app.mainloop()	
#app = App(root)
#root.mainloop()