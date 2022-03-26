from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

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
    http://geomalgorithms.com/a02-_lines.html
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

class LineBuilder(object):
	print("[INJECT][LineBuilder] Triggered...")
	
	epsilon = 30 #in pixels

	def __init__(self, line):
		if line.figure is None:
			raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
		
		print("[INJECT][LineBuilder][__init__] Triggered...")
		
		canvas = line.figure.canvas
		self.canvas = canvas
		self.line = line
		self.axes = line.axes
		self.xs = list(line.get_xdata())
		self.ys = list(line.get_ydata())

		self.ind = None

		canvas.mpl_connect('button_press_event', self.button_press_callback)
		canvas.mpl_connect('button_release_event', self.button_release_callback)
		canvas.mpl_connect('key_press_event', self.key_press_callback)
		canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

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

		self.line.set_animated(True)
		self.canvas.draw()
		self.background = self.canvas.copy_from_bbox(self.line.axes.bbox)

		self.axes.draw_artist(self.line)
		self.canvas.blit(self.axes.bbox)

	def button_release_callback(self, event):
		print("[INJECT][LineBuilder][button_release_callback] Triggered...")
		
		if event.button != 1:
			return
		self.ind = None
		self.line.set_animated(False)
		self.background = None
		self.line.figure.canvas.draw()

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
		self.axes.draw_artist(self.line)
		self.canvas.blit(self.axes.bbox)

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
					self.axes.draw_artist(self.line)
					self.canvas.draw_idle()
					break

if __name__ == '__main__':

    fig, ax = plt.subplots()
    #line = Line2D([0,0.5,1], [0,0.5,1], marker = 'o', markerfacecolor = 'red')
    line = Line2D([0,0.5,1], [0,0.5,1], marker = 'o', color = "#AEAEAE", markerfacecolor = '#009A44')
    ax.add_line(line)

    linebuilder = LineBuilder(line)

    ax.set_title('click to create lines')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    plt.show()