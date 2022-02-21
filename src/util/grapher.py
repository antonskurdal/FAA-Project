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

def dist(x, y):
	"""
	Return the distance between two points.
	"""
	d = x - y
	return np.sqrt(np.dot(d, d))

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

	showverts = True
	epsilon = 5  # max pixel distance to count as a vertex hit

	def __init__(self, ax, poly, obj):
		if poly.figure is None:
			raise RuntimeError('You must first add the polygon to a figure '
	                       'or canvas before defining the interactor')
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
		x = list(self.line.get_xdata())
		x.pop()
		y = list(self.line.get_ydata())
		y.pop()		
		#print(new_data.get_xdata())
		#obj.new_data = self.line
		#self.obj.new_data = Line2D(x, y)
		#self.obj.xs = x
		#self.obj.ys = y
		self.obj.current[self.obj.xs_colname] = x
		self.obj.current[self.obj.ys_colname] = y
	
	def on_key_press(self, event):
		"""Callback for key presses."""
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
		elif event.key == 'i':
			xys = self.poly.get_transform().transform(self.poly.xy)
			p = event.x, event.y  # display coords
			for i in range(len(xys) - 1):
				s0 = xys[i]
				s1 = xys[i + 1]
				d = dist_point_to_segment(p, s0, s1)
				if d <= self.epsilon:
					self.poly.xy = np.insert(
		            self.poly.xy, i+1,
		        [event.xdata, event.ydata],
		        axis=0)
					self.line.set_data(zip(*self.poly.xy))
					break
		if self.line.stale:
			self.canvas.draw_idle()
	
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
	
	print(obj.xs_colname)
	print(obj.ys_colname)
	print(obj.current.index.tolist())
	
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
	#plt.tight_layout()
	
	#obj.xs = xs
	#obj.ys = ys	
	
	#ax.set_xlim((-2, 2))
	#ax.set_ylim((-2, 2))
	#print(type(p))
	
	#return

class ConnectionInteractor:
	"""
	A polygon editor.

	Key-bindings

	  't' toggle vertex markers on and off.  When vertex markers are on,
	      you can move them, delete them

	  'd' delete the vertex under point

	  'i' insert a vertex at point.  You must be within epsilon of the
	      line connecting two existing vertices

	"""

	showverts = True
	epsilon = 5  # max pixel distance to count as a vertex hit

	def __init__(self, ax, poly, obj):
		if poly.figure is None:
			raise RuntimeError('You must first add the polygon to a figure '
	                       'or canvas before defining the interactor')
		self.ax = ax
		canvas = poly.figure.canvas
		self.poly = poly
		obj = obj

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
		x = list(new_data.get_xdata())
		x.pop()
		y = list(new_data.get_ydata())
		y.pop()		
		#print(new_data.get_xdata())
		#obj.new_data = self.line
		obj.new_data = Line2D(x, y)
		return obj
	
	def on_key_press(self, event):
		"""Callback for key presses."""
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
		elif event.key == 'i':
			xys = self.poly.get_transform().transform(self.poly.xy)
			p = event.x, event.y  # display coords
			for i in range(len(xys) - 1):
				s0 = xys[i]
				s1 = xys[i + 1]
				d = dist_point_to_segment(p, s0, s1)
				if d <= self.epsilon:
					self.poly.xy = np.insert(
		            self.poly.xy, i+1,
		        [event.xdata, event.ydata],
		        axis=0)
					self.line.set_data(zip(*self.poly.xy))
					break
		if self.line.stale:
			self.canvas.draw_idle()
	
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


class LineBuilder(object):
	

	epsilon = 30 #in pixels

	def __init__(self, ax, line):
		canvas = line.figure.canvas
		self.line = line
		self.axes = ax
		self.xs = list(line.get_xdata())
		self.ys = list(line.get_ydata())
		
		
		
		self.ind = None
		canvas.mpl_connect('button_press_event', self.button_press_callback)
		canvas.mpl_connect('button_release_event', self.button_release_callback)
		canvas.mpl_connect('key_press_event', self.key_press_callback)
		canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
		self.canvas = canvas

	def get_ind(self, event):
		print("ind")
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
		print("press")
		if event.button != 1:
			return
		if event.inaxes is None:
			return
		self.ind = self.get_ind(event)
		print(self.ind)

		self.line.set_animated(True)
		self.canvas.draw()
		self.background = self.canvas.copy_from_bbox(self.line.axes.bbox)

		self.axes.draw_artist(self.line)
		self.canvas.blit(self.axes.bbox)

	def button_release_callback(self, event):
		print("release")
		if event.button != 1:
			return
		self.ind = None
		self.line.set_animated(False)
		self.background = None
		self.line.figure.canvas.draw()

	def motion_notify_callback(self, event):
		print("motion")
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
		self.axes.draw_artist(self.line)
		self.canvas.blit(self.axes.bbox)

	def key_press_callback(self, event):
		"""Callback for key presses."""
		print("key press")
		if not event.inaxes:
			return
		elif event.key == 'd':
			ind = self.get_ind(event)
			if ind is not None and len(self.xs) > 2:
				self.xs = np.delete(self.xs, ind)
				self.ys = np.delete(self.ys, ind)
				self.line.set_data(self.xs, self.ys)
				self.axes.draw_artist(self.line)
				self.canvas.draw_idle()
		elif event.key == 'i':
			p = np.array([event.x, event.y])  # display coords
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
					self.axes.draw_artist(self.line)
					self.canvas.draw_idle()
					break
		"""Callback for key presses."""

		if not event.inaxes:
			return
		elif event.key == 'd':
			ind = self.get_ind(event)
			if ind is not None and len(self.xs) > 2:
				self.xs = np.delete(self.xs, ind)
				self.ys = np.delete(self.ys, ind)
				self.line.set_data(self.xs, self.ys)
				self.axes.draw_artist(self.line)
				self.canvas.draw_idle()
		elif event.key == 'i':
			p = np.array([event.x, event.y])  # display coords
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
					self.axes.draw_artist(self.line)
					self.canvas.draw_idle()
					break

def plotInteractiveLine(parent, xs, ys, obj):
	
	obj.new_data = Line2D(xs, ys)
	#print(xs)
	#print(obj.xs)
	#print(obj.xs_colname)
	
	poly = ConnectionPatch(np.column_stack([xs, ys]), animated=True, alpha = 0.1)
	
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
	
	ax.add_patch(poly)
	p = ConnectionInteractor(ax, poly, obj)
	
	ax.set_title(obj.xs_colname + " vs " + obj.ys_colname + '\nClick and drag a point to move it')
	ax.set_xlabel(obj.xs_colname)
	ax.set_ylabel(obj.ys_colname)
	ax.autoscale()
	
	#ax.set_xlim((-2, 2))
	#ax.set_ylim((-2, 2))		
	
	return

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