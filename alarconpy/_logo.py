from numpy import *
import matplotlib.pylab as plt
import pkg_resources
import posixpath

def alarconpy_logo():
	try:
		fname = 'alarconpy_logo.npy'
		fpath = posixpath.join('_logos', fname)
	except KeyError:
		raise ValueError('Unknown logo size or selection')

	logo=load(pkg_resources.resource_stream('alarconpy', fpath))

	plt.imshow(logo)
	plt.show() 
