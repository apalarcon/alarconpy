from metpy.units import units
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from alarconpy.cb import *
import metpy.calc as cl
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from math import floor, ceil
import numpy as np
from scipy.ndimage.filters import minimum_filter, maximum_filter
from metpy.plots import add_metpy_logo, Hodograph, SkewT
from metpy.interpolate import interpolate_to_grid, remove_nan_observations
import matplotlib.gridspec as gridspec
import metpy.calc as mpcalc



def extrema_h(mat,mode='wrap',window=10):
    """find the indices of local extrema (min and max)
    in the input array."""
    mn = minimum_filter(mat, size=window, mode=mode)
    mx = maximum_filter(mat, size=window, mode=mode)
    # (mat == mx) true if pixel is equal to the local max
    # (mat == mn) true if pixel is equal to the local in
    # Return the indices of the maxima, minima
    return np.nonzero(mat == mn), np.nonzero(mat == mx)



def readlims(n,filename):
	f = open(filename,"rb")
	print(f)
	v1, v2 = 0.0, 0.0	
	for i in range(n):
		l1, l2 = f.readline().split()
		v1 = float(l1)
		v2 = float(l2)
	return v1, v2	


def plot_wind_h(fig,lon,lat,u,v,dom="d01",ffmin=0,ffmax=270,ffinterval=2,windunits="km/h",barbs="yes"):
	"""Plot wind field

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	u : numpy 2d array
	array of u wind component

	v : numpy 2d array
	array of v wind component

	dom : str
	domain using by file to plot  d01, d02 d03

	ffmin : int
	min value for ff to colorbar

	ffmax : int
	max value for ff to colorbar

	ffmax : int
	max value for ff to colorbar

	ffinterval : int
	interval for colorbar

	windunits : str
	units for wind plot 

	barbs : str
	plot barbs  yes or no
	

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""


	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	
	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6


	ff=cl.wind_speed(u,v)
	cf = ax.contourf(lon, lat, ff, range(int(ffmin),int(ffmax), int(ffinterval)),cmap=colorbarwind(),transform=ccrs.PlateCarree(),extend='both')
	if barbs=="yes":
		ax.barbs(lon[::s,::s], lat[::s,::s], u[::s,::s], v[::s,::s],length=5,sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3),          	linewidth=0.75)
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(windunits, size=17)
	cb.ax.tick_params(labelsize=17)
	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return




def plot_stream_h(fig,lon,lat,u,v,dom,ffmin,ffmax,ffinterval,windunits):
	"""Plot wind field

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	u : numpy 2d array
	array of u wind component

	v : numpy 2d array
	array of v wind component

	dom : str
	domain using by file to plot  d01, d02 d03

	ffmin : int
	min value for ff to colorbar

	ffmax : int
	max value for ff to colorbar

	ffmax : int
	max value for ff to colorbar

	ffinterval : int
	interval for colorbar

	windunits : str
	units for wind plot 
	

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""


	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6


	ff=cl.wind_speed(u,v)
	cf = ax.contourf(lon, lat, ff, range(int(ffmin), int(ffmax), int(ffinterval)),cmap=colorbarwind(),transform=ccrs.PlateCarree(),extend='both')
	ax.streamplot(lon, lat, u, v, transform=ccrs.PlateCarree(), linewidth=1.5, density=1.5, color='yellow')
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(windunits, size=17)
	cb.ax.tick_params(labelsize=17)

	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return






def plot_mslp_h(fig,lon,lat,p,dom="d01",pmin=882,pmax=1020,pinterval=10,punits="hPa",contour_lines="yes"):
	"""Plot mean sea level pressure

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	p : numpy 2d array
	array of mean sea level pressure

	dom : str
	domain using by file to plot  d01, d02 d03

	pmin: int
	pmin for colorbar
	
	pmax:int
	pmax for colorbar

	pinterval: int
	interval for colorbar


	punits : str
	units for mean sea level pressuere plot
	
	contour_lines : str
	yes to plot contour lines and no do not

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""


	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6

	li=np.arange(pmin,pmax,pinterval)
	cf = ax.contourf(lon, lat, p, range(int(pmin),int(pmax),int(pinterval)),cmap=colorbarmslp(),transform=ccrs.PlateCarree(),extend='both')
	if contour_lines=="yes":
		cs = ax.contour(lon, lat, p, transform=ccrs.PlateCarree(),                colors='white', linewidths=1.0, linestyles='solid')
		ax.clabel(cs, fontsize=12, inline=1, inline_spacing=5,    fmt='%i', rightside_up=True, use_clabeltext=True)
	

	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(punits, size=17)
	cb.ax.tick_params(labelsize=17)

	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return





def plot_rain_h(fig,lon,lat,rain,dom="d01",unit="mm"):
	"""Plot rain

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	rain : numpy 2d array
	array of rain

	dom : str
	domain using by file to plot  d01, d02 d03

		
	unit: str
	units to colorbar (rain must be in mm/h)

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""
	
	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6
		

	
	clevs = np.array([0,0.1,0.5,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750])
	
	cf = ax.contourf(lon, lat, rain, clevs,cmap=s3pcpn,transform=ccrs.PlateCarree(),extend='both')
	
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(unit, size=17)
	cb.ax.tick_params(labelsize=17)

	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return










def plot_tv_h(fig,lon,lat,u,v,p,rain,dom="d01",unit="mm"):
	"""Plot rain

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com

	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	u: numpy 2d array
	array of u wind component

	v: numpy 2d array
	array of v wind component

	p: numpy 2d array
	array of sea level pressure

	rain: numpy 2d array
	array of rain

	dom : str
	domain using by file to plot  d01, d02 d03

		
	unit: str
	units for rain to colorbar

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""
	
	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.background_img(name='BM', resolution='high')
	ax.add_feature(cfeature.STATES, linewidth=0.6)
	#ax.coastlines(color='yellow',resolution='10m',linewidth=0.6)
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)

	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6


	clevs = np.array([0.5,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750])
	
	cf = ax.contourf(lon, lat, rain, levels=clevs[:len(clevs)], vmin=-5, vmax=clevs[-1],cmap=s3pcpn,transform=ccrs.PlateCarree())
	
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(unit, size=17)
	cb.ax.tick_params(labelsize=17)

	
	cs = ax.contour(lon, lat, p, transform=ccrs.PlateCarree(),                colors='yellow', linewidths=1.0, linestyles='solid')
	ax.clabel(cs, fontsize=12, inline=1, inline_spacing=5,    fmt='%i', rightside_up=True, use_clabeltext=True)

	
	ax.barbs(lon[::s,::s], lat[::s,::s], u[::s,::s], v[::s,::s],length=5,sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3),        	linewidth=0.75,color="white")


	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}


	transformed = crs.transform_points(ccrs.Geodetic(), lon, lat)
	x = transformed[..., 0]  # lons in projection coordinates
	y = transformed[..., 1]  # lats in projection coordinates
	local_min, local_max = extrema(p, window=50)
	xlows = x[local_min]; xhighs = x[local_max]
	ylows = y[local_min]; yhighs = y[local_max]
	lowvals = p[local_min]; highvals = p[local_max]

	xyplotted = []


	xmin, xmax, ymin, ymax = ax.get_extent()
	yoffset = 0.022*(ymax-ymin)
	dmin = yoffset
	for x,y,p in zip(xlows, ylows, lowvals):
		if x < xmax and x > xmin and y < ymax and y > ymin:
			dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
			if not dist or min(dist) > dmin:
				plt.text(x,y,'B',fontsize=17,fontweight='bold',ha='center',va='center',color='r')
	   # mapa.scatter(x,y,s=40,color='white')
            #plt.text(x,y-yoffset,repr(int(p)),fontsize=9,
                 #   ha='center',va='bottom',color='r')
                   # bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
				xyplotted.append((x,y))
# plot highs as red H's, with max pressure value underneath.
	xyplotted = []
	for x,y,p in zip(xhighs, yhighs, highvals):
		if x < xmax and x > xmin and y < ymax and y > ymin:
			dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
			if not dist or min(dist) > dmin:
				plt.text(x,y,'A',fontsize=17,fontweight='bold',ha='center',va='center',color='b')
            #plt.text(x,y-yoffset,repr(int(p)),fontsize=9,
                   # ha='center',va='bottom',color='b')
                    #bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
				xyplotted.append((x,y))
	return




def plot_windarea_h(fig,lon,lat,wind,dom="d02",ffmin=0,ffmax=270,ffinterval=2,windunits="km/h"):
	"""Plot wind field

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	wind : numpy 2d array
	array of wind speed

	
	dom : str
	domain using by file to plot  d01, d02 d03

	ffmin : int
	min value for ff to colorbar

	ffmax : int
	max value for ff to colorbar

	ffmax : int
	max value for ff to colorbar

	ffinterval : int
	interval for colorbar

	windunits : str
	units for wind plot 

		

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""


	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	
	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6


	
	cf = ax.contourf(lon, lat, wind, range(int(ffmin),int(ffmax), int(ffinterval)),cmap=colorbarwind(),transform=ccrs.PlateCarree(),extend='both')
	
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(windunits, size=17)
	cb.ax.tick_params(labelsize=17)

	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return


def plot_rainarea_h(fig,lon,lat,rainarea,dom="d02",unit="mm"):
	"""Plot rain

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com

	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	rainarea : numpy 2d array
	array of rain

	dom : str
	domain using by file to plot  d01, d02 d03

		
	unit: str
	units to colorbar (rain must be in mm/h)

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""
	
	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6
		

	
	clevs = np.array([0,0.1,0.5,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750])
	
	cf = ax.contourf(lon, lat, rainarea, clevs,cmap=s3pcpn,transform=ccrs.PlateCarree(),extend='both')
	
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(unit, size=17)
	cb.ax.tick_params(labelsize=17)

	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return


def plot_geop_wind(fig,lon,lat,u,v,geop,dom="d01",ffmin=0,ffmax=270,ffinterval=2,windunits="km/h"):
	"""Plot geoptotential height + wind speed

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	u : numpy 2d array
	array of u wind component

	v : numpy 2d array
	array of v wind component

	geop : numpy 2d array
	array of geoptotential height

	dom : str
	domain using by file to plot  d01, d02 d03

	ffmin : int
	min value for ff to colorbar

	ffmax : int
	max value for ff to colorbar

	ffmax : int
	max value for ff to colorbar

	ffinterval : int
	interval for colorbar

	windunits : str
	units for wind plot 

	barbs : str
	plot barbs  yes or no
	

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""


	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	
	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6


	ff=cl.wind_speed(u,v)
	cf = ax.contourf(lon, lat, ff, range(int(ffmin),int(ffmax), int(ffinterval)),cmap=colorbarwind(),transform=ccrs.PlateCarree(),extend='both')
	cs = ax.contour(lon, lat, geop, transform=ccrs.PlateCarree(),                colors='k', linewidths=1.5, linestyles='solid')
	ax.clabel(cs, fontsize=14, inline=1, inline_spacing=5,    fmt='%i', rightside_up=True, use_clabeltext=True)
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(windunits, size=17)
	cb.ax.tick_params(labelsize=17)
	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return


def plot_qwbs(fig,lon,lat,qwbs,dom="d01",qwbsmin=-300,qwbsmax=400,qwbsinterval=20,qwbsunits="$W/m^{2}$"):
	"""Plot Instantaneous latent heat flux

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	qwbs : numpy 2d array
	array of Instantaneous latent heat flux

	dom : str
	domain using by file to plot  d01, d02 d03

	pmin: int
	pmin for colorbar
	
	pmax:int
	pmax for colorbar

	pinterval: int
	interval for colorbar


	punits : str
	units for mean sea level pressuere plot
	
	contour_lines : str
	yes to plot contour lines and no do not

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""


	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6

	li=np.arange(qwbsmin,qwbsmax,qwbsinterval)
	cf = ax.contourf(lon, lat, qwbs, range(int(qwbsmin),int(qwbsmax),int(qwbsinterval)),cmap=plt.cm.jet,transform=ccrs.PlateCarree(),extend='both')
	
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(qwbsunits, size=17)
	cb.ax.tick_params(labelsize=17)

	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return



def plot_RV(fig,lon,lat,rv,dom="d01",rvmin=-3,rvmax=3,rvinterval=0.000001,rvunits="s^{-1}$"):
	"""Plot relative vorticity

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	qwbs : numpy 2d array
	array of relative vorticity

	dom : str
	domain using by file to plot  d01, d02 d03

	pmin: int
	pmin for colorbar
	
	pmax:int
	pmax for colorbar

	pinterval: int
	interval for colorbar


	punits : str
	units for mean sea level pressuere plot
	
	contour_lines : str
	yes to plot contour lines and no do not

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""


	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6

	li=np.arange(rvmin,rvmax,rvinterval)
	cf = ax.contourf(lon, lat, rv, li,cmap=plt.cm.bwr,transform=ccrs.PlateCarree(),extend='both')
	
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(rvunits, size=17)
	cb.ax.tick_params(labelsize=17)

	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return


def plot_temperature(fig,lon,lat,T,dom="d01",li=range(0, 38, 1),Tunits="C",level="superficie"):
	"""Plot temperature field

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	T : numpy 2d array
	array of Temeparture

	dom : str
	domain using by file to plot  d01, d02 d03

	
	Tunits : str
	units for temperature plot 
	
	level: str
	level to plot temperature 
	level Superficie 850 700 500 200

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""


	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)

	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6
		
	cf = ax.contourf(lon, lat, T, li,cmap=plt.cm.jet,transform=ccrs.PlateCarree(),extend='both')
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=30,shrink=0.6,pad=0.06)
	cb.set_label(Tunits, size=17)
	cb.ax.tick_params(labelsize=17)

	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return

def plot_specific_humidity(fig,lon,lat,q,dom="d01",hrmin=0,hrmax=40,hrinterval=5,units="g/kg"):
	"""Plot specific_humidity

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting
	lon : numpy 2d array
	array of longitude

	lat : numpy 2d array
	array of latitude

	q : numpy 2d array
	array of specific humidity

	dom : str
	domain using by file to plot  d01, d02 d03

	hrmin: int
	hrmin for colorbar
	
	hrmax:int
	hrmax for colorbar

	hrinterval: int
	interval for colorbar
	
	contour_lines : str
	add contour line to plot


	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""
	
	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	if dom=="d01":
		ax.set_extent([lon.min()+10,lon.max()-10,lat.min()+7.5,lat.max()-2], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=7
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='vertical'
		s=6
		

	li=np.arange(int(hrmin),int(hrmax),int(hrinterval))
	cf = ax.contourf(lon, lat, q, li,cmap=plt.cm.terrain_r,transform=ccrs.PlateCarree(),extend='both')
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label(units, size=17)
	cb.ax.tick_params(labelsize=17)

	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': 17, 'color': 'black'}
	gl.ylabel_style = {'size': 17,'color': 'black'}
	return
