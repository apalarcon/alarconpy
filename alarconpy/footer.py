from numpy import *
import matplotlib.pylab as plt
import pkg_resources
import posixpath

def add_footer(text_1="",text_2="",text_3="",fontsize=13):
	"""Add footer to a figure.

	Add text footer to a figure.


	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------

	text_1 : str
	text_2 : str
	text_3 : str
	text to put in the figure, all text are lower center	

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""



	
	coords = plt.gca().get_position().get_points()
	
	plt.figtext( (coords[0,0]+1.)/2.0, 0.09, text_1,fontsize=fontsize,horizontalalignment='center',fontweight='bold')
	plt.figtext( (coords[0,0]+1.)/2.0, 0.07, text_2 , horizontalalignment='center', verticalalignment='center',fontsize=fontsize)

	plt.figtext( (coords[0,0]+1.)/2.0, 0.03, text_3, horizontalalignment='center', verticalalignment='center',fontsize=fontsize)
	return


def add_header(text_1="",text_2="",text_3="",fontsize=15):
	"""
	Add text footer to a figure.



	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------

	text_1 : str
	text_2 : str
	text_3 : str


	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""
	
	
	coords = plt.gca().get_position().get_points()
	plt.figtext( (coords[0,0]+0.50)/2.0, 0.97, text_1, horizontalalignment='center', verticalalignment='center',fontsize=fontsize,fontweight="bold")


	plt.figtext( (coords[0,0]+1.5)/2.0, 0.97, text_2, horizontalalignment='center', verticalalignment='center',fontsize=fontsize,fontweight="bold")

	plt.figtext( (coords[0,0]+0.44)/2.0, 0.93, text_3, horizontalalignment='center', verticalalignment='center',fontsize=fontsize-2)
	return



def add_logos(position="lower_left",scale=0.15,x_c=0.5,y_c=0.5):

	"""Add logo to a figure.

	Adds an image  to a figure.


	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com

	Parameters
	----------
	
	postition : str
	postition_logo = lower_left lower_right upper_left  upper_right customize
	scale: float
	Scale of figure values betwen 0-1. default=0.15
	x_c: float
	X position por customize option
	y_c : float
	Y position for customize option
	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""

	try:
		fname = 'all_logos.npy'
		fpath = posixpath.join('_logos', fname)
	except KeyError:
		raise ValueError('Unknown logo size or selection')

	#logo = imread(pkg_resources.resource_stream('metpy.plots', fpath))



	logo=load(pkg_resources.resource_stream('alarconpy', fpath))
	coords = plt.gca().get_position().get_points()

	if position=="lower_right":
		ax = plt.axes([0.84, -0.030, scale, scale])
	elif position=="lower_left":
		ax = plt.axes([0.005, -0.03, scale, scale])
	elif position=="upper_left":
		ax = plt.axes([0.01, 0.88,scale, scale])
	elif position=="upper_right":
		ax = plt.axes([0.84, 0.88, scale, scale])
	elif position=="customize":
		if x_c==None or y_c==None:
			raise ValueError('Please set x and y position')
		else:
			ax = plt.axes([x_c, y_c, scale, scale])
	else:
		raise ValueError('Please set a position equal to lower_right, lower_left, upper_right, upper_left,customize')
	

	ax.set_xticks([])
	ax.set_yticks([])
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)

	ax.imshow(logo)
	
	return


def add_customize_logo(logo="instec",position="lower_left",scale=0.1,x_c=0.5,y_c=0.5):

	"""Add logo to a figure.

	Adds an image  to a figure.

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com

	Parameters
	----------
	logo: array
	image in npy array
	default logos=ecna, instec,met,fama,nthf

	postition : str
	postition_logo = lower_left lower_right upper_left  upper_right
	scale: float
	Scale of figure values betwen 0-1. default=0.1
	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""
	if logo=="ecna":
		try:
			fname = 'ecna.npy'
			fpath = posixpath.join('_logos', fname)
		except KeyError:
			raise ValueError('Unknown logo size or selection')

		logo=load(pkg_resources.resource_stream('alarconpy', fpath))
	elif logo=="instec":
		try:
			fname = 'instec.npy'
			fpath = posixpath.join('_logos', fname)
		except KeyError:
			raise ValueError('Unknown logo size or selection')

		logo=load(pkg_resources.resource_stream('alarconpy', fpath))

	elif logo=="fama":
		try:
			fname = 'fama.npy'
			fpath = posixpath.join('_logos', fname)
		except KeyError:
			raise ValueError('Unknown logo size or selection')

		logo=load(pkg_resources.resource_stream('alarconpy', fpath))
	elif logo=="met":
		try:
			fname = 'met.npy'
			fpath = posixpath.join('_logos', fname)
		except KeyError:
			raise ValueError('Unknown logo size or selection')
	elif logo=="nthf":
		try:
			fname = 'nthf.npy'
			fpath = posixpath.join('_logos', fname)
		except KeyError:
			raise ValueError('Unknown logo size or selection')

		logo=load(pkg_resources.resource_stream('alarconpy', fpath))
	elif logo=="legend":
		try:
			fname = 'legend.npy'
			fpath = posixpath.join('_logos', fname)
		except KeyError:
			raise ValueError('Unknown logo size or selection')

		logo=load(pkg_resources.resource_stream('alarconpy', fpath))
	
	
	coords = plt.gca().get_position().get_points()

	if position=="lower_right":
		ay = plt.axes([0.88, 0.010, scale, scale])
	elif position=="lower_left":
		ay = plt.axes([0.01, 0.008, scale, scale])
	elif position=="upper_left":
		ay = plt.axes([0.01, 0.9, scale, scale])
	elif position=="upper_right":
		ay = plt.axes([0.88, 0.9, scale, scale])
	elif position=="center_left":
		ay = plt.axes([-0.09, 0.4, scale, scale])
	elif position=="customize":
		if x_c==None or y_c==None:
			raise ValueError('Please set x and y position')
		else:
			ay = plt.axes([x_c, y_c, scale, scale])
	else:
		raise ValueError('Please set a position equal to lower_right, lower_left, upper_right, upper_left,customize')

	ay.set_xticks([])
	ay.set_yticks([])
	ay.spines['right'].set_visible(False)
	ay.spines['left'].set_visible(False)
	ay.spines['top'].set_visible(False)
	ay.spines['bottom'].set_visible(False)

	ay.imshow(logo)
	
	return




def add_logos_inst(position="lower_left",scale=0.25,x_c=0.5,y_c=0.5):

	"""Add logo to a figure.

	Adds an image  to a figure.

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com

	Parameters
	----------
	
	postition : str
	postition_logo = lower_left lower_right upper_left  upper_right
	scale: float
	Scale of figure values betwen 0-1. default=0.25
	
	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""

	try:
		fname = 'logosinst.npy'
		fpath = posixpath.join('_logos', fname)
	except KeyError:
		raise ValueError('Unknown logo size or selection')

	#logo = imread(pkg_resources.resource_stream('metpy.plots', fpath))



	logo=load(pkg_resources.resource_stream('alarconpy', fpath))
	coords = plt.gca().get_position().get_points()

	if position=="lower_right":
		az = plt.axes([0.72, -0.05, scale, scale])
	elif position=="lower_left":
		az = plt.axes([0.01, -0.05, scale, scale])
	elif position=="upper_left":
		az = plt.axes([0.01, 0.85, scale, scale])
	elif position=="upper_right":
		az = plt.axes([0.72, 0.85, scale, scale])
	elif position=="customize":
		if x_c==None or y_c==None:
			raise ValueError('Please set x and y position')
		else:
			az = plt.axes([x_c, y_c, scale, scale])
	else:
		raise ValueError('Please set a position equal to lower_right, lower_left, upper_right, upper_left,customize')

	az.set_xticks([])
	az.set_yticks([])
	az.spines['right'].set_visible(False)
	az.spines['left'].set_visible(False)
	az.spines['top'].set_visible(False)
	az.spines['bottom'].set_visible(False)

	az.imshow(logo)
	
	return
