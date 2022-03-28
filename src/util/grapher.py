"""
############################

06/28/2021

Developers: Anton Skurdal


Description:

############################
"""

###########
# IMPORTS #
###########
import numpy as np

import tkinter as tk
from tkinter import*

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.patches import Polygon
from matplotlib.patches import ConnectionPatch
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 

import time
import pandas as pd


def dist(x, y):
	"""
	Return the distance between two points.
	"""
	d = x - y
	return np.sqrt(np.dot(d, d))


def dist_point_to_segment(p, s0, s1):
	"""
	Get the distance of a point to a segment.
	  *p*, *s0*, *s1* are *xy* sequences
	This algorithm from
	http://www.geomalgorithms.com/algorithms.html
	"""
	v = s1 - s0
	w = p - s0
	c1 = np.dot(w, v)
	if c1 <= 0:
		return dist(p, s0)
	c2 = np.dot(v, v)
	if c2 <= c1:
		return dist(p, s1)
	b = c1 / c2
	pb = s0 + b * v
	return dist(p, pb)


def curve_fit(x, y):
	#Create a line of evenly spaced numbers over the interval len(x)
	
	x_fitLine = np.linspace(x[0], x[-1], num = len(x) * 10)

	#Fit the line and save the coefficients
	coefs = np.polyfit(x, y, 9)

	#Use the values of x_fitLine as inputs for the polynomial
	#function with given coefficients to create y_fitLine
	y_fitLine = np.polyval(x_fitLine, coefs)

	#Plot the fitLine
	#plt.plot(x_fitLine, y_fitLine)
	
	#printEquation(coefs)
	
	#plt.show()
	

	return coefs, x, y	

class PolygonInteractor:
	"""
	A polygon editor.

	Key-bindings

	  't' toggle vertex markers on and off.  When vertex markers are on,
		  you can move them, delete them

	  'd' delete the vertex under point

	  'i' insert a vertex at point.  You must be within epsilon of the
		  line connecting two existing vertices

	"""
	print("[INJECT][PolygonInteractor] Triggered...")
	
	showverts = True
	epsilon = 5  # max pixel distance to count as a vertex hit

	def __init__(self, ax, poly, obj):
		print("[INJECT][PolygonInteractor][__init__] Triggered...")
		
		if poly.figure is None:
			raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
			
		self.ax = ax
		canvas = poly.figure.canvas
		self.poly = poly
		self.obj = obj

		x, y = zip(*self.poly.xy)
		self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
		self.ax.add_line(self.line)
		
		self.cid = self.poly.add_callback(self.poly_changed)
		self._ind = None  # the active vert

		canvas.mpl_connect('draw_event', self.on_draw)
		canvas.mpl_connect('button_press_event', self.on_button_press)
		canvas.mpl_connect('key_press_event', self.on_key_press)
		canvas.mpl_connect('button_release_event', self.on_button_release)
		canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
		self.canvas = canvas

	def on_draw(self, event):
		self.background = self.canvas.copy_from_bbox(self.ax.bbox)
		self.ax.draw_artist(self.poly)
		self.ax.draw_artist(self.line)
		# do not need to blit here, this will fire before the screen is
		# updated
		print("")

	def poly_changed(self, poly):
		"""This method is called whenever the pathpatch object is called."""
		# only copy the artist props to the line (except visibility)
		vis = self.line.get_visible()
		Artist.update_from(self.line, poly)
		self.line.set_visible(vis)  # don't use the poly visibility state

	def get_ind_under_point(self, event):
		"""
		Return the index of the point closest to the event position or *None*
		if no point is within ``self.epsilon`` to the event position.
		"""
		# display coords
		xy = np.asarray(self.poly.xy)
		xyt = self.poly.get_transform().transform(xy)
		xt, yt = xyt[:, 0], xyt[:, 1]
		d = np.hypot(xt - event.x, yt - event.y)
		indseq, = np.nonzero(d == d.min())
		ind = indseq[0]

		if d[ind] >= self.epsilon:
			ind = None
		
		return ind

	def on_button_press(self, event):
		"""Callback for mouse button presses."""
		print("[INJECT][PolygonInteractor][on_button_press] Triggered...")
		
		
		if not self.showverts:
			return
		if event.inaxes is None:
			return
		if event.button != 1:
			return
		self._ind = self.get_ind_under_point(event)

	def on_button_release(self, event):
		"""Callback for mouse button releases."""
		if not self.showverts:
			return
		if event.button != 1:
			return
		self._ind = None
		new_data = self.line
		x = list(self.line.get_xdata()[:-1])
		#x.pop()
		y = list(self.line.get_ydata()[:-1])
		#y.pop()		
		
		
		#print(new_data.get_xdata())
		#obj.new_data = self.line
		#self.obj.new_data = Line2D(x, y)
		#self.obj.xs = x
		#self.obj.ys = y
		self.obj.current[self.obj.xs_colname] = x
		self.obj.current[self.obj.ys_colname] = y
	
	def on_key_press(self, event):
		"""Callback for key presses."""
		key_error = False
		print("[INJECT][GRAPHER][POLYGONINTERACTOR] polyxy len BEFORE:{}".format(len(self.poly.xy)))
		#print("[INJECT][GRAPHER][POLYGONINTERACTOR] polyxy BEFORE:\n{}".format(self.poly.xy))
		
		
		if not event.inaxes:
			return
		if event.key == 't':
			self.showverts = not self.showverts
			self.line.set_visible(self.showverts)
			if not self.showverts:
				self._ind = None
		elif event.key == 'd':
			ind = self.get_ind_under_point(event)
			if ind is not None:
				self.poly.xy = np.delete(self.poly.xy,
								 ind, axis=0)
				self.line.set_data(zip(*self.poly.xy))
		
		
		
		
		###################################################################################################
		elif event.key == 'i':
			print("\n========== 'I' Pressed ==========")
			#print("[INJECT][GRAPHER][POLYGONINTERACTOR] polyxy len before:{}".format(len(self.poly.xy)))
			
			
			xys = self.poly.get_transform().transform(self.poly.xy)
			p = event.x, event.y  # display coords
			
			
			
			#print("[INJECT][GRAPHER][POLYGONINTERACTOR] p: {}".format(p))

			
			
			
			prev_data = zip(*self.poly.xy)
			#print("[INJECT][GRAPHER][POLYGONINTERACTOR] polyxy len after:{}".format(len(self.poly.xy)))
			#print(self.poly.xy)
			
			
			for i in range(len(xys) - 1):
				#print("[INJECT][GRAPHER][POLYGONINTERACTOR][LOOP {}] polyxy len TEST:{}".format(i, len(self.poly.xy)))
				
				s0 = xys[i]
				s1 = xys[i + 1]
				d = dist_point_to_segment(p, s0, s1)
				
				# Check if insert location is close enough to the line
				if d <= self.epsilon:
					
					print("[INJECT][GRAPHER][POLYGONINTERACTOR] event.x, event.y: {}".format([event.xdata, event.ydata]))
					
					time.sleep(1)
					
					self.poly.xy = np.insert(self.poly.xy, i+1, [event.xdata, event.ydata], axis=0)
				
					#print("[INJECT][GRAPHER][POLYGONINTERACTOR] polyxy len2:\n{}".format(len(self.poly.xy)))
				
					self.line.set_data(zip(*self.poly.xy))
					
					testdf = self.obj.current
					xcol = self.obj.xs_colname
					ycol = self.obj.ys_colname
					
					# print("[INJECT][GRAPHER][POLYGONINTERACTOR] testdf:\n{}".format(testdf))
					# print("[INJECT][GRAPHER][POLYGONINTERACTOR] line xdata:\n{}".format(self.line.get_xdata()))
					# print("[INJECT][GRAPHER][POLYGONINTERACTOR] line ydata:\n{}".format(self.line.get_ydata()))
					# print("[INJECT][GRAPHER][POLYGONINTERACTOR] line xlen: {}".format(len(list(self.line.get_xdata()))))
					
					# CURRENT PROBLEM IS THAT WHEN POINTS ARE ADDED BEFORE MIN AND/OR AFTER MAX,
					# THEY NEED TO NOT BE APPLIED. THIS ISN'T TOUGH, BUT THE INDEX GETS MESSED UP AFTER
					# IT HAPPENS WHICH MAKES THE USER UNABLE TO ADD MORE VALID POINTS.
					# THE USER COULD JUST BE TRUSTED, BUT IT WOULD BE ANNOYING WHEN WORKING WITH
					# OVERLAPPING LINES OR MISCLICKS.
					
					x = list(self.line.get_xdata()[:-1])
					y = list(self.line.get_ydata()[:-1])
					
					#print(x)
					#print("\n")
					
					try:
						
						i = 0
						while i < len(x):
							
							if(testdf[xcol][i] != x[i] or testdf[ycol][i] != y[i]):
								#print("testdf[xcol][i]: {:.2f}\tx[i]: {:.2f}\tindex: {} [different]".format(testdf[xcol][i], x[i], i))
								
								
								#first = testdf.iloc[0: i]
								#print(first[xcol])
								
								testdf = pd.concat([testdf.iloc[0: i],  pd.DataFrame({xcol:x[i], ycol:y[i]}, index=[i+1]), testdf.loc[i:]], ignore_index=True)
								
								#print(new[xcol])
								
								#time.sleep(100)
								
								#print(x[i], testdf[xcol][i])
								#print("different. i = " + str(i))
								""" testdf.iloc[i] = 0
								print(testdf)
								time.sleep(10)
								testdf.at[i, xcol] = x[i]
								testdf.at[i, ycol] = y[i]
								
								
								print(testdf[i]) """
								#print(testdf[xcol][i], x[i])
								#print(list(testdf[xcol]))
								
								
							else:
								#print("testdf[xcol][i]: {:.2f}\tx[i]: {:.2f}\tindex: {} [same]".format(testdf[xcol][i], x[i], i))
								pass
								#print("same. i = " + str(i))
								
								#print(testdf[i])
							i = i + 1
					except KeyError:
						print("KeyError")
						self.line.set_data(prev_data)
						key_error = True
						#break
						
						#print("[INJECT][GRAPHER][POLYGONINTERACTOR] testdf[xcol][i]:{}\tline_x[xcol][i]:{}".format(testdf[xcol][i]), x[i])
						#if(testdf[xcol][i] != x_new[i] or testdf[xcol][i] != x_new[i]):
						#	print("Different")
						
					self.obj.current = testdf
					#print(self.obj.current[xcol])
					
					#testdf[self.obj.xs_colname] = list(self.line.get_xdata())
					#testdf[self.obj.ys_colname] = list(self.line.get_ydata())
					
					
					#line = DataFrame({"onset": 30.0, "length": 1.3}, index=[3])
					#df2 = concat([df.iloc[:2], line, df.iloc[2:]]).reset_index(drop=True)
					#self.obj.current[self.obj.xs_colname] = list(self.line.get_xdata())
					#self.obj.current[self.obj.ys_colname] = list(self.line.get_ydata())
					
					#break
		
		print("[INJECT][GRAPHER][POLYGONINTERACTOR] polyxy len AFTER:{}".format(len(self.poly.xy)))
		#print("[INJECT][GRAPHER][POLYGONINTERACTOR] polyxy AFTER:\n{}".format(self.poly.xy))
								
		#time.sleep(5)
				
		if self.line.stale and key_error == False:
			print("[INJECT] LINE IS STALE!")
			self.canvas.draw_idle()
		
		""" if self.line.stale:
			self.canvas.draw_idle() """
		###################################################################################################
		
	
	def on_mouse_move(self, event):
		"""Callback for mouse movements."""
		if not self.showverts:
			return
		if self._ind is None:
			return
		if event.inaxes is None:
			return
		if event.button != 1:
			return
		x, y = event.xdata, event.ydata

		self.poly.xy[self._ind] = x, y
		if self._ind == 0:
			self.poly.xy[-1] = x, y
		elif self._ind == len(self.poly.xy) - 1:
			self.poly.xy[0] = x, y
		self.line.set_data(zip(*self.poly.xy))
		
		self.canvas.restore_region(self.background)
		self.ax.draw_artist(self.poly)
		self.ax.draw_artist(self.line)
		self.canvas.blit(self.ax.bbox)


def plotInteractivePolygon(master, obj):
	
	#print(obj.xs_colname)
	#print(obj.ys_colname)
	#print(obj.current.index.tolist())
	
	#Set xs and ys, make sure index works
	if(obj.xs_colname == "index"):
		xs = obj.current.index.tolist()
	else:
		xs = obj.current[obj.xs_colname]
	
	if(obj.ys_colname == "index"):
		ys = obj.current.index.tolist()
	else:
		ys =  obj.current[obj.ys_colname]
	
	
	
	poly = Polygon(np.column_stack([xs, ys]), animated=True, alpha = 0.1)
	#poly = Polygon(np.column_stack([obj.current[obj.xs_colname], obj.current[obj.ys_colname]]), animated=True, alpha = 0.1)
	
	fig = Figure()
	ax = fig.add_subplot(111)
	
	#Place graph
	canvas = FigureCanvasTkAgg(fig, master)
	canvas.draw()
	canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")
	
	toolbarFrame = Frame(master=master)
	toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
	toolbarFrame.grid_rowconfigure(0, weight = 1)
	toolbarFrame.grid_columnconfigure(0, weight = 1)
	
	toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
	toolbar.grid(row = 0, column = 0, sticky="NSEW")				
	#print(obj.xs)
	#print(obj.ys)	
	
	ax.add_patch(poly)
	p = PolygonInteractor(ax, poly, obj)
	
	ax.set_title(obj.xs_colname + " vs " + obj.ys_colname + '\nClick and drag a point to move it')
	ax.set_xlabel(obj.xs_colname)
	ax.set_ylabel(obj.ys_colname)
	ax.autoscale()



class LineBuilder:
	print("[INJECT][LineBuilder] Triggered...")
	
	epsilon = 30 #in pixels

	def __init__(self, ax, line, obj):
		print("[INJECT][LineBuilder][__init__] Triggered...")
		if line.figure is None:
			raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
		
		self.ax = ax
		canvas = line.figure.canvas
		self.line = line
		self.obj = obj
		
		x = list(line.get_xdata())
		y = list(line.get_ydata())
		
		
		self.line = Line2D(x, y, marker='o', markerfacecolor='pink', animated=True)
		
		self.ax.add_line(self.line)
		
		self.oldx = x
		self.oldy = y
		
		
		self.cid = self.line.add_callback(self.line_changed)
		self._ind = None # the active vert
		
		
		#canvas = line.figure.canvas
		self.canvas = canvas
		#self.line = line
		#self.axes = line.axes
		self.xs = list(line.get_xdata())
		self.ys = list(line.get_ydata())

		self.ind = None
		
		canvas.mpl_connect('draw_event', self.on_draw)
		canvas.mpl_connect('button_press_event', self.button_press_callback)
		canvas.mpl_connect('button_release_event', self.button_release_callback)
		canvas.mpl_connect('key_press_event', self.key_press_callback)
		canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
		self.canvas = canvas
		
		
		
		
	def line_changed(self, line):
		print("[INJECT][LineBuilder][line_changed] Triggered...")
		
		"""This method is called whenever the pathpatch object is called."""
		# only copy the artist props to the line (except visibility)
		vis = self.line.get_visible()
		#Artist.update_from(self.line, line)
		self.line.set_visible(vis)  # don't use the poly visibility state
	
	def on_draw(self, event):
		print("[INJECT][LineBuilder][on_draw] Triggered...")
		self.background = self.canvas.copy_from_bbox(self.ax.bbox)
		self.ax.draw_artist(self.line)
		# do not need to blit here, this will fire before the screen is
		# updated
		print("")

	def get_ind(self, event):
		print("[INJECT][LineBuilder][get_ind] Triggered...")
		
		xy = np.asarray(self.line._xy)
		xyt = self.line.get_transform().transform(xy)
		x, y = xyt[:, 0], xyt[:, 1]
		d = np.sqrt((x-event.x)**2 + (y - event.y)**2)
		indseq, = np.nonzero(d == d.min())
		ind = indseq[0]

		if d[ind] >= self.epsilon:
			ind = None

		return ind

	def button_press_callback(self, event):
		print("[INJECT][LineBuilder][button_press_callback] Triggered...")
		
		if event.button != 1:
			return
		if event.inaxes is None:
			return
		self.ind = self.get_ind(event)
		print(self.ind)
		
		
		# self.line.set_animated(True)
		# self.canvas.draw()
		# self.background = self.canvas.copy_from_bbox(self.line.axes.bbox)

		self.ax.draw_artist(self.line)
		# self.canvas.blit(self.ax.bbox)

	def button_release_callback(self, event):
		print("[INJECT][LineBuilder][button_release_callback] Triggered...")
		
		if event.button != 1:
			return
		self.ind = None
		
		self.line.set_animated(True)
		self.background = None
		
		self.line.figure.canvas.draw()
		
		print("OLDX = {}".format(self.oldx))
		print("X = {}".format(self.line.get_xdata()))
		
		

	def motion_notify_callback(self, event):
		#print("[INJECT][LineBuilder][motion_notify_callback] Triggered...")
		
		if event.inaxes != self.line.axes:
			return
		if event.button != 1:
			return
		if self.ind is None:
			return
		self.xs[self.ind] = event.xdata
		self.ys[self.ind] = event.ydata
		self.line.set_data(self.xs, self.ys)

		self.canvas.restore_region(self.background)
		self.ax.draw_artist(self.line)
		self.canvas.blit(self.ax.bbox)
		

	def key_press_callback(self, event):
		"""Callback for key presses."""
		print("[INJECT][LineBuilder][key_press_callback] Triggered...")
		
		if not event.inaxes:
			return
		elif event.key == 'd':
			print("\tKey pressed = 'D'")
			ind = self.get_ind(event)
			if ind is not None and len(self.xs) > 2:
				self.xs = np.delete(self.xs, ind)
				self.ys = np.delete(self.ys, ind)
				self.line.set_data(self.xs, self.ys)
				self.axes.draw_artist(self.line)
				self.canvas.draw_idle()
		elif event.key == 'i':
			print("\tKey pressed = 'I'")
			
			p = np.array([event.x, event.y])  # display coords
			print("p = {}".format(p))
			
			xy = np.asarray(self.line._xy)
			xyt = self.line.get_transform().transform(xy)
			for i in range(len(xyt) - 1):
				s0 = xyt[i]
				s1 = xyt[i+1]
				d = dist_point_to_segment(p, s0, s1)
				if d <= self.epsilon:
					self.xs = np.insert(self.xs, i+1, event.xdata)
					self.ys = np.insert(self.ys, i+1, event.ydata)
					self.line.set_data(self.xs, self.ys)
					self.ax.draw_artist(self.line)
					self.canvas.draw_idle()
					break
	
	""" def on_mouse_move(self, event):
		#Callback for mouse movements.
		print("[INJECT][LineBuilder][on_mouse_move] Triggered...")
			
		if not self.showverts:
			return
		if self._ind is None:
			return
		if event.inaxes is None:
			return
		if event.button != 1:
			return
		x, y = event.xdata, event.ydata

		self.poly.xy[self._ind] = x, y
		if self._ind == 0:
			self.poly.xy[-1] = x, y
		elif self._ind == len(self.poly.xy) - 1:
			self.poly.xy[0] = x, y
		self.line.set_data(zip(*self.poly.xy))
		
		self.canvas.restore_region(self.background)
		self.ax.draw_artist(self.poly)
		self.ax.draw_artist(self.line)
		self.canvas.blit(self.ax.bbox) """







def plotInteractiveLine(parent, obj):
	
	#fig, ax = plt.subplots()
	#line = Line2D([0,0.5,1], [0,0.5,1], marker = 'o', markerfacecolor = 'red')
	
	#line = Line2D([0,0.5,1], [0,0.5,1], animated = True)
	
	line = Line2D([0,0.5,1], [0,0.5,1], marker = 'o', color = "#AEAEAE", markerfacecolor = '#009A44', animated = True)
	#line = Line2D([0,0.5,1], [0,0.5,1], marker = 'o', color = "white", markerfacecolor = 'white')
	
	
	
	#fig, ax = plt.subplots()
	
	fig = Figure()
	ax = fig.add_subplot(111)
	
	
	# Place graph
	canvas = FigureCanvasTkAgg(fig, parent)
	canvas.draw()
	canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")
	
	toolbarFrame = Frame(parent)
	toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
	toolbarFrame.grid_rowconfigure(0, weight = 1)
	toolbarFrame.grid_columnconfigure(0, weight = 1)
	
	toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
	
	from pathlib import Path
	#print(Path.cwd())
	import util.sku_widgets as sku
	#button = Button(master = toolbar, text = "hello", command = lambda: print("yeet"), width = 20)
	
	tab_controller = parent.master.master
	def reset_plot():
			obj.current = obj.base.copy(deep = True)
			
	button = sku.BorderButton(master = toolbar, button_text = "Reset Plot", button_command = lambda: [reset_plot(), plotInteractiveLine(parent, obj)], button_activebackground="#009A44")
	button.child['width'] = 20
	
	
	
	button.pack(side = "left", fill = "both", padx = (2, 2), pady = (8, 8))
	#button.pack_propagate(False)
	
	
	toolbar.grid(row = 0, column = 0, sticky="NSEW")
	toolbar.pack_propagate(False)
	toolbarFrame.grid_propagate(False)
	
	#print(toolbar.toolitems)
	
	
	# tm = fig.canvas.manager.toolmanager
	# tm.add_tool("newtool", NewTool)
	# fig.canvas.manager.toolbar.add_tool(tm.get_tool("newtool"), "toolgroup")
	

	ax.add_line(line)

	linebuilder = LineBuilder(ax, line, obj)

	ax.set_title('click to create lines')
	ax.set_xlim(-2,2)
	ax.set_ylim(-2,2)
	
	
	
	
	#plt.show()


# def plotInteractiveLine(parent, obj):
	
# 	print("\n[INJECT][plotInteractiveLine] Starting...")
# 	#Set xs and ys, make sure index works
# 	if(obj.xs_colname == "index"):
# 		xs = obj.current.index.tolist()
# 	else:
# 		xs = obj.current[obj.xs_colname]
	
# 	if(obj.ys_colname == "index"):
# 		ys = obj.current.index.tolist()
# 	else:
# 		ys =  obj.current[obj.ys_colname]
	
# 	line = Line2D(xs, ys, marker = 'o', color = "#AEAEAE", markerfacecolor = '#009A44')
	
# 	fig = Figure()
# 	ax = fig.add_subplot(111)
	
	
# 	#Place graph
# 	canvas = FigureCanvasTkAgg(fig, parent)
# 	canvas.draw()
# 	canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")
	
# 	toolbarFrame = Frame(parent)
# 	toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
# 	toolbarFrame.grid_rowconfigure(0, weight = 1)
# 	toolbarFrame.grid_columnconfigure(0, weight = 1)
	
# 	toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
# 	toolbar.grid(row = 0, column = 0, sticky="NSEW")
	
# 	ax.add_line(line)
# 	linebuilder = LineBuilder(ax, line, obj)
	
	
	
	
# 	ax.set_title(obj.xs_colname + " vs " + obj.ys_colname + '\nClick and drag a point to move it')
# 	ax.set_xlabel(obj.xs_colname)
# 	ax.set_ylabel(obj.ys_colname)
# 	ax.autoscale()
	
	
















def plot_basic(master, x, y, xlabel, ylabel):

	fig = Figure()
	ax = fig.add_subplot(111)
	
	#Place graph
	canvas = FigureCanvasTkAgg(fig, master)
	canvas.draw()
	canvas.get_tk_widget().grid(row = 0, column = 0, sticky = "NSEW")
	
	toolbarFrame = Frame(master=master)
	toolbarFrame.grid(row=1,column=0, sticky = "NSEW", padx=(0,0), pady=(0,0))
	toolbarFrame.grid_rowconfigure(0, weight = 1)
	toolbarFrame.grid_columnconfigure(0, weight = 1)
	
	toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
	toolbar.grid(row = 0, column = 0, sticky="NSEW")
	
	ax.scatter(x, y)
	ax.autoscale()
	ax.set_title(xlabel + " vs " + ylabel)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	#ax.set_xlim((-2, 2))
	#ax.set_ylim((-2, 2))
	
	return