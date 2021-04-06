from metpy.units import units
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from alarconpy.cb import *
from alarconpy.calc import wind_rose_dir
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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def extrema(mat,mode='wrap',window=10):
    """find the indices of local extrema (min and max)
    in the input array."""
    mn = minimum_filter(mat, size=window, mode=mode)
    mx = maximum_filter(mat, size=window, mode=mode)
    # (mat == mx) true if pixel is equal to the local max
    # (mat == mn) true if pixel is equal to the local in
    # Return the indices of the maxima, minima
    return np.nonzero(mat == mn), np.nonzero(mat == mx)



def plot_wind(fig,lon,lat,u,v,dom="d01",ffmin=0,ffmax=300,ffinterval=2,windunits="km/h",barbs="yes"):
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
		ax.set_extent([lon.min()+3,lon.max()-3,lat.min()+1,lat.max()-1], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=5
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+0.2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='horizontal'
		s=6
	if dom=="d03":
		ax.set_extent([lon.min()+0.1,lon.max()-0.1,lat.min()+0.1,lat.max()-0.1], crs=ccrs.PlateCarree())
		paso_h=1
		cbposition='horizontal'
		s=9


	ff=cl.wind_speed(u,v)
	cf = ax.contourf(lon, lat, ff, range(int(ffmin), int(ffmax), int(ffinterval)),cmap=colorbarwind(),transform=ccrs.PlateCarree(),extend='both')
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




def plot_stream(fig,lon,lat,u,v,dom="d01",ffmin=0,ffmax=300,ffinterval=2,windunits="km/h"):
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
		ax.set_extent([lon.min()+3,lon.max()-3,lat.min()+1,lat.max()-1], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=8
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+0.2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='horizontal'
		s=9
	if dom=="d03":
		ax.set_extent([lon.min()+0.1,lon.max()-0.1,lat.min()+0.1,lat.max()-0.1], crs=ccrs.PlateCarree())
		paso_h=1
		cbposition='horizontal'
		s=12


	ff=cl.wind_speed(u,v)
	cf = ax.contourf(lon, lat, ff, range(int(ffmin), int(ffmax), int(ffinterval)),cmap=colorbarwind(),transform=ccrs.PlateCarree(),extend='both')
	ax.streamplot(lon, lat, u, v, transform=ccrs.PlateCarree(), linewidth=1.5, density=1.5, color='yellow')
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



def plot_temperature(fig,lon,lat,T,dom="d01",Tunits="C",level="superficie"):
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
		ax.set_extent([lon.min()+3,lon.max()-3,lat.min()+1,lat.max()-1], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+0.2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='horizontal'
		
	if dom=="d03":
		ax.set_extent([lon.min()+0.1,lon.max()-0.1,lat.min()+0.1,lat.max()-0.1], crs=ccrs.PlateCarree())
		paso_h=1
		cbposition='horizontal'
		


	if level == "Superficie" or level == "superficie":
		li=range(0, 38, 1)
	if level == "850":
		li=range(1, 27, 1)
	if level == "700":
		li=range(-4, 15, 1)
	if level == "500":
		li=range(-20, -2, 1)
	if level == "200":
		li=range(-75, -45, 1)
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





def plot_mslp(fig,lon,lat,p,dom="d01",pmin=970,pmax=1020,pinterval=5,punits="hPa",contour_lines="yes"):
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
		ax.set_extent([lon.min()+3,lon.max()-3,lat.min()+1,lat.max()-1], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+0.2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='horizontal'
		
	if dom=="d03":
		ax.set_extent([lon.min()+0.1,lon.max()-0.1,lat.min()+0.1,lat.max()-0.1], crs=ccrs.PlateCarree())
		paso_h=1
		cbposition='horizontal'
		

	li=np.arange(int(pmin),int(pmax),int(pinterval))
	cf = ax.contourf(lon, lat, p, li,cmap=colorbarmslp(),transform=ccrs.PlateCarree(),extend='both')
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



def plot_relative_humidity(fig,lon,lat,hr,dom="d01",hrmin=0,hrmax=100,hrinterval=5,contour_lines="no"):
	"""Plot relative_humidity

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

	hr : numpy 2d array
	array of relative humidity

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
	hr[hr<0]=0
	hr[hr>100]=100

	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
	ax.add_feature(cfeature.STATES, linewidth=0.6)


	if dom=="d01":
		ax.set_extent([lon.min()+3,lon.max()-3,lat.min()+1,lat.max()-1], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+0.2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='horizontal'
		
	if dom=="d03":
		ax.set_extent([lon.min()+0.1,lon.max()-0.1,lat.min()+0.1,lat.max()-0.1], crs=ccrs.PlateCarree())
		paso_h=1
		cbposition='horizontal'
		

	li=np.arange(int(hrmin),int(hrmax),int(hrinterval))
	cf = ax.contourf(lon, lat, hr, li,cmap=plt.cm.terrain_r,transform=ccrs.PlateCarree(),extend='both')
	if contour_lines=="yes":
		cs = ax.contour(lon, lat, hr, transform=ccrs.PlateCarree(),                colors='white', linewidths=1.0, linestyles='solid')
		ax.clabel(cs, fontsize=12, inline=1, inline_spacing=5,    fmt='%i', rightside_up=True, use_clabeltext=True)
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label("%", size=17)
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





def plot_rain(fig,lon,lat,rain,dom="d01",unit="mm"):
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
		ax.set_extent([lon.min()+3,lon.max()-3,lat.min()+1,lat.max()-1], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+0.2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='horizontal'
		
	if dom=="d03":
		ax.set_extent([lon.min()+0.1,lon.max()-0.1,lat.min()+0.1,lat.max()-0.1], crs=ccrs.PlateCarree())
		paso_h=1
		cbposition='horizontal'
		

	
	clevs = np.array([0,0.1,0.5,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750])
	cmap,norm,lvs=colorbarprecip()
	
	#cf = ax.contourf(lon, lat, rain, clevs,cmap=s3pcpn,transform=ccrs.PlateCarree(),extend='both')
	cf = ax.contourf(lon, lat, rain,lvs,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),extend='both')
	
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




def plot_radar(fig,lon,lat,reflect,dom="d01",unit="dBZ"):
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

	relectividad : numpy 2d array
	array of rain

	dom : str
	domain using by file to plot  d01, d02 d03

	

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
		ax.set_extent([lon.min()+3,lon.max()-3,lat.min()+1,lat.max()-1], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+0.2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='horizontal'
		
	if dom=="d03":
		ax.set_extent([lon.min()+0.1,lon.max()-0.1,lat.min()+0.1,lat.max()-0.1], crs=ccrs.PlateCarree())
		paso_h=1
		cbposition='horizontal'
		

	
	clevs = np.array([0,1,2.5,5,7.5,10,15,20,30,40,50,70,80,90])
	
	cf = ax.contourf(lon, lat, reflect, clevs,cmap=colorbaradar(),transform=ccrs.PlateCarree(),extend='both')
	
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





def plot_nubes(fig,lon,lat,nubes,dom="d01",unit="%"):
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

	nubes: numpy 2d array
	array of nubes

	dom : str
	domain using by file to plot  d01, d02 d03

	rmin: int
	rmin for colorbar
	
	rmax:int
	rmax for colorbar

	rinterval: int
	interval for colorbar
	
	unit: str
	units to colorbar

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""
	
	crs = ccrs.PlateCarree()
	ax=plt.subplot(111,projection=ccrs.PlateCarree())
	
	ax.add_feature(cfeature.STATES, linewidth=0.6)
	ax.coastlines(color='yellow',resolution='10m',linewidth=0.6)
	#ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)

	if dom=="d01":
		ax.set_extent([lon.min()+3,lon.max()-3,lat.min()+1,lat.max()-1], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+0.2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='horizontal'
		
	if dom=="d03":
		ax.set_extent([lon.min()+0.1,lon.max()-0.1,lat.min()+0.1,lat.max()-0.1], crs=ccrs.PlateCarree())
		paso_h=1
		cbposition='horizontal'
		

	#li=np.linspace(rmin,rmax,rinterval)
	li=np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])	
	cf = ax.contourf(lon, lat, nubes, li,cmap=plt.cm.binary_r,transform=ccrs.PlateCarree(),extend='both')
	
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





def plot_tv(fig,lon,lat,u,v,p,rain,dom="d01",unit=""):
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
		ax.set_extent([lon.min()+3,lon.max()-3,lat.min()+1,lat.max()-1], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=5
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+0.2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='horizontal'
		s=6
	if dom=="d03":
		ax.set_extent([lon.min()+0.1,lon.max()-0.1,lat.min()+0.1,lat.max()-0.1], crs=ccrs.PlateCarree())
		paso_h=1
		cbposition='horizontal'
		s=9


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




def ntwf_sounding(fig,T,Td,p,u,v,wind_speed,wind_dir,st="",station_name="",stationlat="",stationlon="",title="",text_0="",text_1=""):
	"""Plot sounding

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com

	Parameters
	----------
	fig : `matplotlib.figure`
	The `figure` instance used for plotting

	T: numpy 1d array
	array of air temperature

	Td : numpy 1d array
	array of dew temperature

	p: numpy 1d array
	array of pressure

	u: numpy 1d array
	array of u wind component

	v: numpy 1d array
	array of v wind component

	wind_speed: numpy 1d array
	array of wind_speed


	wind_dir: numpy 1d array
	array of wind_dir

	st: str
	stattion code

	station_name : str
	name of station

	stationlat : float
	lat of station

	stationlon : float
	lon of station

	
		
	title: str
	title of figure, located on upper center
	
	text_0: str
	located on lower right


	text_1: str
	located on upper right

	Returns
	-------
	`matplotlib.image.FigureImage`
	The `matplotlib.image.FigureImage` instance created

	"""



	gs = gridspec.GridSpec(3, 3)

	skew = SkewT(fig, rotation=45, subplot=gs[:, :2])
	T=T*units.celsius
	Td=Td*units.celsius
	p=p*units.hPa
	# Plot the data using normal plotting functions, in this case using
	# log scaling in Y, as dictated by the typical meteorological plot
	skew.plot(p, T, 'r',linewidth=3)
	skew.plot(p, Td, 'g',linewidth=3)
	skew.plot_barbs(p, u, v)
	skew.ax.set_ylim(1000, 100)
	skew.ax.set_xlim(-40, 60)
	
	plt.text(-29,950,"0.4",color="black")
	plt.text(-18,950,"1",color="black")
	plt.text(-9,950,"2",color="black")
	plt.text(0,950,"4",color="black")
	plt.text(8,950,"7",color="black")
	plt.text(12,950,"10",color="black")
	plt.text(21,950,"16",color="black")
	plt.text(27,950,"24",color="black")
	plt.text(32,950,"32",color="black")
	plt.text(35,950,"g/kg",color="black")

	# Calculate LCL height and plot as black dot
	lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
	skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

	# Calculate full parcel profile and add to plot as black line
	prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
	skew.plot(p, prof, 'k', linewidth=2)

	# Shade areas of CAPE and CIN
	skew.shade_cin(p, T, prof)
	skew.shade_cape(p, T, prof)
	

	text="Estación: "+station_name+"\nLat:"+str(stationlat)+", Lon: "+str(stationlon)
	#text="Estación "+str(station_name)

	# An example of a slanted line at constant T -- in this case the 0
	# isotherm
	skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

	# Add the relevant special lines
	skew.plot_dry_adiabats()
	skew.plot_moist_adiabats()
	skew.plot_mixing_lines()
	
	plt.title(title)
	plt.xlabel("Temperatura (ºC)")
	plt.ylabel("Niveles de Presión (hPa)")
	# Good bounds for aspect ratio
	skew.ax.set_xlim(-30, 40)

	# Create a hodograph
	ax = fig.add_subplot(gs[1, -1])
	h = Hodograph(ax, component_range=60.)
	h.add_grid(increment=20)
	h.plot(u, v)
	plt.xlabel("nudos")

	
	crs = ccrs.LambertConformal(central_longitude=-80.0, central_latitude=22.5)
	ayy=plt.subplot(333,projection=crs)
	ayy.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.75)
	ayy.add_feature(cfeature.STATES, linewidth=0.5)
	ayy.set_extent([-85,-74,19,24], crs=ccrs.PlateCarree())
	#ayy.stock_img()
	ayy.add_feature(cfeature.LAND,color="#bfbfbf") #If I comment this => all ok, but I need 
	ayy.add_feature(cfeature.OCEAN,color="#b8dffe")
	ayy.add_feature(cfeature.LAKES)
	ayy.add_feature(cfeature.RIVERS)
	ayy.scatter(stationlon,stationlat,marker='.',color="r",transform=ccrs.Geodetic())
	plt.text(stationlon+0.7,stationlat-0.3,st[2:5],color="black",backgroundcolor="white",fontsize="8",fontweight="bold",transform=ccrs.Geodetic())



	coords = plt.gca().get_position().get_points()

	plt.figtext( (coords[0,0]+1.3)/2.0, 0.11, text_0, horizontalalignment='right', verticalalignment='center',fontsize="7")

	plt.figtext((coords[0,0]+0.75)/2.0, 0.22, text, bbox=dict(facecolor='none', edgecolor='black'),horizontalalignment='left', verticalalignment='center',fontsize="11")

	
	plt.figtext( (coords[0,0]+1.3)/2.0, 0.958, text_1, horizontalalignment='right', verticalalignment='center',fontsize="10")





def plot_gdi_index(fig,lon,lat,gdi,dom="d01"):
	"""Plot gdi

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

	gdi : numpy 2d array
	array of gdi values

	dom : str
	domain using by file to plot  d01, d02 d03

	

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
		ax.set_extent([lon.min()+3,lon.max()-3,lat.min()+1,lat.max()-1], crs=ccrs.PlateCarree())
		paso_h=5
		cbposition='vertical'
		s=5
	if dom=="d02":
		ax.set_extent([lon.min()+0.5,lon.max()-0.5,lat.min()+0.2,lat.max()-0.2], crs=ccrs.PlateCarree())
		paso_h=2
		cbposition='horizontal'
		s=6
	if dom=="d03":
		ax.set_extent([lon.min()+0.1,lon.max()-0.1,lat.min()+0.1,lat.max()-0.1], crs=ccrs.PlateCarree())
		paso_h=1
		cbposition='horizontal'
		s=9

	colors=["#191919","#303030","#383838","#525252","#5a5a5a","#787878","#808080","#7996a2",
                "#63a5b6","#41be70","#50ca61","#c8c839","#d2d23e","#f0a030","#f08c35","#f03246","#dc0a3c","#96005a",]

	lvs=[-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,55,65]
	cmap,norm=matplotlib.colors.from_levels_and_colors(lvs,colors,extend="both")
	cf = ax.contourf(lon, lat, gdi,cmap=cmap,norm=norm,levels=lvs,transform=ccrs.PlateCarree(),extend='both')
	
	cb = fig.colorbar(cf, orientation=cbposition, extend='both', aspect=40,shrink=0.6,pad=0.06)
	cb.set_label("GDI", size=17)
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


def saeta(xc, yc, ang, r, dx, color1, color2, fig):
    import numpy as np
    import matplotlib.pyplot as plt
    
    da=np.arctan(3/(5*r/dx+ 2))
    x = [r*np.cos(ang)+xc, np.sqrt(r**2+6*r*dx/5+13/25*dx**2)*np.cos(ang+da)+xc, (r-dx)*np.cos(ang)+xc, np.sqrt(r**2+6*r*dx/5+13/25*dx**2)*np.cos(ang-da)+xc, r*np.cos(ang)+xc]
    y = [r*np.sin(ang)+yc, np.sqrt(r**2+6*r*dx/5+13/25*dx**2)*np.sin(ang+da)+yc, (r-dx)*np.sin(ang)+yc, np.sqrt(r**2+6*r*dx/5+13/25*dx**2)*np.sin(ang-da)+yc, r*np.sin(ang)+yc]
    
    plt.fill(x, y, facecolor=color1, edgecolor= color2, lw=0.5)

def numero(x, y, dx, dy):
    
	plt.plot([x-dx, x-dx], [y+dy, y-dy], color='orange')
	plt.plot([x+dx, x+dx], [y+dy, y-dy], color='orange')

	plt.plot([x-dy, x+dy], [y-dx, y-dx], color='orange')
	plt.plot([x-dy, x+dy], [y+dx, y+dx], color='orange')

def meteoplot(dd, ff,head,ti=0):
    """
    To get wind meteogram

    Author: Alexander Lobaina LaO and adapted to alarconpy by Albenis Pérez Alarcón
    contact:apalarcon1991@gmail.com

    Parameters
    -----------------
    dd:array (N,)
    wind direction in degrees

    ff:array (N,)
    wind speed in kt

    ti:int
    hour init to forecast
    """
    dd = np.mod((9 -np.round(dd/10)),36)*np.pi/18

    ff = np.round(ff, 0)
    ff1 = []
    for i in range (len(ff)):
        ff1.append('%i'%(ff[i]))

  
    dic = dict() 
    dic.__setitem__('06',[0, 0.13])
    dic.__setitem__('MID',[0, 0])
    dic.__setitem__('24',[0.13, 0])
  
    dlt=dic[head]
  
    colordata='r'#(0, .9, .1)
    cardinales = ['E', 'N', 'W', 'S']
    tm = ['00 Z', '01 Z', '02 Z', '03 Z', '04 Z', '05 Z', '06 Z', '07 Z', '08 Z', '09 Z', '10 Z', '11 Z',
          '12 Z', '13 Z', '14 Z', '15 Z', '16 Z', '17 Z', '18 Z', '19 Z', '20 Z', '21 Z', '22 Z', '23 Z']

    orientacion = np.pi/6

    a = np.arange(36)*np.pi/18 #direccion del viento cada 10 grados
    b = np.arange(4)*np.pi/2   #cuadrantes
    p = [orientacion, orientacion+np.pi]   #orientacion de la pista

    fig = plt.figure(figsize=(20, 13))
    plt.axis('off')
    plt.fill([-7, 72, 72,-7], [-11,-11, 46, 46], facecolor='w',edgecolor='c')
    #LEGENDA
    saeta(-5., -9., 0., 1, 0.8, (0.95, 0.95, 0.9), 'b', fig)
    saeta(24., -9., 0., 1, 0.8, colordata,'k', fig)
    
    plt.plot([25, 25.3], [-8.95, -8.95], color=colordata,linewidth=1)
    numero(51, -9, 0.3, 0.7)

    plt.text(-3, -9, 'ORIENTACIÓN EN SALTOS 10º', verticalalignment = 'center', color='#6f7174',fontsize =15, alpha=50,fontweight="bold")
    plt.text(26, -9, 'DIRECCIÓN DEL VIENTO (º)', verticalalignment = 'center', color=colordata,fontsize =15, alpha=50,fontweight="bold")
    plt.text(53, -9, 'FUERZA DEL VIENTO (KT)', verticalalignment = 'center', color='orange',fontsize =15, alpha=50,fontweight="bold")
    



    for i in np.arange(6):
        for j in np.arange(4):

            for ang in a:
                saeta(13*i, 13*j, ang, 5, 0.6, 'w', '#6f7174', fig)
            k=0
            for ang in b:
                saeta(13*i, 13*j, ang, 5, 0.6, 'w', 'b', fig)
                plt.text(13*i+ 4*np.cos(ang), 13*j+ 4*np.sin(ang), cardinales[k], color='b', horizontalalignment = 'center', verticalalignment = 'center', fontsize=10, alpha=3,fontweight="bold")
                k+=1

            rl= [dlt[0]+2.88, dlt[1]+2.88]
            plt.plot(13*i+rl*np.cos(p), 13*j+rl*np.sin(p), 'b',linewidth=24)
            plt.plot(13*i+3.*np.cos(p), 13*j+3.*np.sin(p), '-', color=(0.15, 0.15, 0.15), linewidth=21)

            plt.plot(13*i+4.*np.cos(p), 13*j+4.*np.sin(p), 'w--',linewidth=0.8)
            
            for ang in p:
                ang_air = np.mod(27-int(18*ang/np.pi), 36)               
                cabecera = str(ang_air)
                    
                if ang_air < 10:
                    cabecera = '0'+str(int(ang_air))
                plt.text(13*i+ 6.2*np.cos(ang), 13*j+ 6.2*np.sin(ang), cabecera, color='k', horizontalalignment = 'center', verticalalignment = 'center',fontsize=15)

            plt.plot([13*i+ 5*np.cos(dd[6*(3-j)+i]), 13*i+ 5.3*np.cos(dd[6*(3-j)+i])], [13*j+ 5*np.sin(dd[6*(3-j)+i]), 13*j+ 5.3*np.sin(dd[6*(3-j)+i])], color=colordata,linewidth=1)
            saeta(13*i, 13*j, dd[6*(3-j)+i], 5, 0.6, colordata, 'r', fig)


            plt.text(13*i, 13*(3 - j), str(ff1[6*j+i]), horizontalalignment = 'center', verticalalignment = 'center',
                    color='orange', fontsize=18, alpha=3,fontweight="bold")

            plt.text(13*i, 13*(3-j)-6.5, tm[np.mod(ti+6*j+i, 24)], horizontalalignment = 'center', color='k',fontsize=15)

    return




def plot_gramet(x_cord=np.array([None]),y_cord=np.array([None]),
				u=np.array([None]),v=np.array([None]),T=np.array([None]),
				ff=np.array([None]),dd=np.array([None]),llws=np.array([None]),
				T2=np.array([None]),dewT=np.array([None]),psfc=np.array([None])
				,mslp=np.array([None]),plazos=np.array([None]),figsize=(18,12),
				title="",figlayout="yes",fontsize=15,windbarbs="yes",wind_isoline="yes",T_max=28,T_min=15,ff_min=15,llws_min=8):
	"""
	Plot meteogram
	
	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	-------------------------------------
	x_cord: array like (N,)
	y_cord: array like (D,)
	
	u,v,T array for u wind component in (kt), v wind component (kt) and T temperautre (C) like (D,N)
	
	ff,dd,llw,T2,dewT,psfc,mslp array like (N,) for wind speed (ff in kt), wind direction (dd in degree), low level wind shear (llws kt), temperature at 2m (T2 celsius), Dew temperature point (dewT in celsius), surface pressure (psfc in hpa) and mean sea level pressure (mlsp in hPa)
	
	plazos=array like (N,)
	Set hours in UTC Ex ["00z", "01Z"...]
	
	figsize=tuple like (w,h)
	to define figure size
	
	figlayout:str  aviable options yes or no
	to use figlayout to plot
	
	fontsize: float, to define fontsize
	
	title: str
	To define plot title
	
	windbarbs: str aviable options yes or no
	to plot windbarbs or no
	
	wind_isoline:str aviable options yes or no
	to plot wind isolines
	
	T_max: float: to set a maximum temperature to plot using warning color (red)
	
	T_min: float to set a minimum temperature to plot using warning color (blue)
	
	ff_min: float to set a minimun wind speed to use warning colors
	
	llw_min:float to set minimun low level jet speed to use warning colors
	
	return
	"""
	
	

	vars_plot=[]
	if ff.any()!=None:
		vars_plot=np.append(vars_plot,"ff")
	if dd.any()!=None:
		vars_plot=np.append(vars_plot,"rose_dd")
		rose_dd=[]
		for dd_ in dd:
			rose_dd=np.append(rose_dd,wind_rose_dir(dd_))
	if llws.any()!=None:
		vars_plot=np.append(vars_plot,"llws")
	if T2.any()!=None:
		vars_plot=np.append(vars_plot,"T2")
	if dewT.any()!=None:
		vars_plot=np.append(vars_plot,"dewT")
	if psfc.any()!=None:
		vars_plot=np.append(vars_plot,"psfc")
	if mslp.any()!=None:
		vars_plot=np.append(vars_plot,"mslp")
	
	fig = plt.figure(figsize=figsize)
	


	gridspec.GridSpec(4,5)
	host =plt.subplot2grid((4,5), (0,0), colspan=5, rowspan=3)
	par1 = host.twinx()
	
	
	
	
	if y_cord.all()==None:
		y_cord=np.arange(1000,50,-50)
	if x_cord.all()==None:
		x_cord=np.arange(0,25,1)
	if plazos==None:
		time=np.array(["00Z","01Z","02Z","03Z","04Z","05Z","06Z","07Z","08Z","09Z","10Z", "11Z", "12Z", "13Z", "14Z", "15Z", "16Z", "17Z", "18Z", "19Z", "20Z", "21Z", "22Z", "23Z", "00Z"])
	else:
		time=plazos
		
	par1.set_ylim(y_cord.max(),y_cord.min())

	host.set_ylim(y_cord.max(),y_cord.min())
	host.set_yticks(np.append(1020,y_cord))
	host.set_yticklabels(np.append(1020,y_cord),fontsize=fontsize)
	par1.set_yticks(np.append(1020,y_cord))
	par1.set_yticklabels(np.append(1020,y_cord),fontsize=fontsize)
	host.set_xticks(x_cord)
	host.set_xticklabels(time,fontsize=fontsize)
	



	host.grid()

	host.set_xlabel("Horas",fontsize=fontsize)
	host.set_ylabel("Altitud (hPa)",fontsize=fontsize)


	if T.any()!=None:
		cs=host.contour(x_cord, y_cord ,T, levels=None,colors='red', linewidths=1.5, linestyles='-')
		host.clabel(cs,fontsize=18, inline=2, inline_spacing=10,    fmt='%i', rightside_up=True, use_clabeltext=True)
		check="yes"
		host.plot([1],[1],label="Temperatura (C)",color="red")

		
	if u.any()!=None and v.any()!=None:
		host.plot([1],[1],label="Velocidad del viento (kt)",color="blue")
		check='yes'
		if wind_isoline=="yes" or  wind_isoline=="Yes" or  wind_isoline=="Y" or  wind_isoline=="y":
			cs=host.contour(x_cord, y_cord ,np.sqrt(v**2 + v**2), colors='blue', linewidths=1.0, linestyles='--')
			host.clabel(cs, fontsize=fontsize+3, inline=2, inline_spacing=10,    fmt='%i', rightside_up=True, use_clabeltext=True)
		
		if windbarbs=="yes" or  windbarbs=="Yes" or  windbarbs=="Y" or  windbarbs=="y":
			host.barbs(x_cord, y_cord, u, v,length=5,sizes=dict(emptybarb=0.35, spacing=0.3, height=0.4),linewidth=0.75)

	if u.any()==None and v.any()==None and T.any()==None:
		check="no"
	
	
	if check=="yes":
		host.legend(loc='lower left',fontsize=fontsize)
	host.set_title(title,fontsize=fontsize+3)



	# small subplot 1
	sfc=plt.subplot2grid((4,5), (3,0),colspan=5, rowspan=1)
	sfc.set_ylim(len(vars_plot)+0.5,0)
	sfc.set_yticklabels(np.arange(len(vars_plot)+1,0,1),fontsize=1)

	sfc.set_xticks(x_cord)
	sfc.set_xticklabels(time,fontsize=fontsize)
	sfc.grid()
	sfc.spines['right'].set_visible(False)
	sfc.spines['left'].set_visible(False)



	T2_color=[]
	dewT_color=[]
	ff_color=[]
	rose_dd_color=[]
	llws_color=[]
	mslp_color=[]
	psfc_color=[]
	
	
	T2_txt=[]
	dewT_txt=[]
	ff_txt=[]
	rose_dd_txt=[]
	llws_txt=[]
	mslp_txt=[]
	psfc_txt=[]
	
	

		

	for i in range(0,len(time)):
		
		if T2.any()!=None:
			if T2[i]>T_max:
				T2_color=np.append(T2_color,"red")
				T2_txt=np.append(T2_txt,"white")
			elif T2[i]<T_min:
				T2_color=np.append(T2_color,"blue")
				T2_txt=np.append(T2_txt,"white")
			else:
				T2_color=np.append(T2_color,"silver")
				T2_txt=np.append(T2_txt,"black")
				
				
		if dewT.any()!=None:
			dewT_color=np.append(dewT_color,"silver")
			dewT_txt=np.append(dewT_txt,"black")
				
		
		if ff.any()!=None:
			if ff[i]>ff_min:
				ff_color=np.append(ff_color,"red")
				ff_txt=np.append(ff_txt,"white")
			else:
				ff_color=np.append(ff_color,"silver")
				ff_txt=np.append(ff_txt,"black")
				
				
		if dd.any()!=None:
			if 90<dd[i]<180 or 270<dd[i]<359:
				rose_dd_color=np.append(rose_dd_color,"red")
				rose_dd_txt=np.append(rose_dd_txt,"white")
			else:
				rose_dd_color=np.append(rose_dd_color,"silver")
				rose_dd_txt=np.append(rose_dd_txt,"black")
		
		if llws.any()!=None:
			if llws[i]>llws_min:
				llws_color=np.append(llws_color,"red")
				llws_txt=np.append(llws_txt,"white")
			else:
				llws_color=np.append(llws_color,"silver")
				llws_txt=np.append(llws_txt,"black")
			
		if psfc.any()!=None:
			psfc_color=np.append(psfc_color,"silver")
			psfc_txt=np.append(psfc_txt,"black")
			
		if mslp.any()!=None:
			mslp_color=np.append(mslp_color,"silver")
			mslp_txt=np.append(mslp_txt,"black")



	sfc.text(0,-0.05,"En superficie",fontsize=fontsize)

	for i in range(0,len(vars_plot)):
		if vars_plot[i]=="rose_dd":
			sfc.text(-1.2,i+1,"dd",fontsize=fontsize,fontweight="bold")
		else:
			sfc.text(-1.2,i+1,vars_plot[i],fontsize=fontsize,fontweight="bold")

	
	for i in range(0,len(x_cord)):
		for j in range(0,len(vars_plot)):
			if vars_plot[j]=="rose_dd":
				
				sfc.text(x_cord[i]-0.15,j+1,vars()[vars_plot[j]][i],fontsize=13,backgroundcolor=vars()[vars_plot[j]+"_color"][i],color=vars()[vars_plot[j]+"_txt"][i],fontweight="bold")
			else:
				sfc.text(x_cord[i]-0.15,j+1,int(vars()[vars_plot[j]][i]),fontsize=13,backgroundcolor=vars()[vars_plot[j]+"_color"][i],color=vars()[vars_plot[j]+"_txt"][i],fontweight="bold")

		
	
	
	
	fig.tight_layout()
	
	return

def imscatter(x, y, image, ax=None, zoom=1):
	"""
	Customize marker from image
	
	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	-------------------------------------
	x: x-direction coordinates
	y: y-direction coordinates
	image: path to image
	zoom: image scale to plot
	
	retunr
	a fig marker
	"""
	
	
	
	if ax is None:
		ax = plt.gca()
	try:
		image = plt.imread(image)
	except TypeError:
		# Likely already an array...
		pass
	im = OffsetImage(image, zoom=zoom)
	x, y = np.atleast_1d(x, y)
	artists = []
	for x0, y0 in zip(x, y):
		ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
		artists.append(ax.add_artist(ab))
	ax.update_datalim(np.column_stack([x, y]))
	#ax.autoscale()
	return artists

