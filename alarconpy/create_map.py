"""
Autor: Albenis Pérez Alarcón
Last Update: abril 19, 2019
apalarcon1991@gmail.com
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import sys
import os
from alarconpy.paths import *
import numpy as np
from math import floor, ceil
os.environ["CARTOPY_USER_BACKGROUNDS"]=cartopy_BM()


def get_map(lower_left_corner=(-85,19),upper_right_corner=(-73,30),dlon=None,bg="None",res="medium",cr="10m",landcolor="#bfbfbf",oceancolor="#b8dffe",fontsize=15,id_=111):
	"""
	To create a map with Cartopy
	
	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
		
	Parameters
	-------------
	
	
	lower_left_corner: coordinate like (lon,lat)
	
	upper_right_corner: coordinate like (lon,lat)
	
	bg: string to define map background_img
	
	aviable options
	None: to use  empty map
	BM: to use Bluemarble background, aviable resolutions: low, medium, high, full
	product : to use vegetation background, aviable resolutions:low, high
	topo: to use topography background, aviable resolutions: low, high
	stock: default cartopy background, use default resolution
	define_color:to set specific colors to land and ocean define by landcolor and oceancolor 
	
	res: string to get background resolution
	cr: string Coast resolution
	aviable options: 110m or 10m
	
	fontsize:float
	to set fontsize to plot draw_labels
	
	id_: float
	position in figure (111 is default to plot in all figure)
	Return a created map
	
	"""
	
	
	min_lon,min_lat=lower_left_corner
	max_lon,max_lat=upper_right_corner
	
	crs = ccrs.PlateCarree()
	mapa=plt.subplot(id_,projection=ccrs.PlateCarree())
	if cr=="110m" or cr=="10m":
		mapa.add_feature(cfeature.COASTLINE.with_scale(cr), linewidth=1)
	else:
		raise ValueError('Aviable coast resolution are "110m" and "10m"')
	mapa.add_feature(cfeature.STATES, linewidth=0.25)
	mapa.set_extent([min_lon,max_lon,min_lat,max_lat], crs=ccrs.PlateCarree())
	
	if dlon==None:
		if abs(min_lon-max_lon)<=2:
			paso_h=0.5
		elif 3<abs(min_lon-max_lon)<=8:
			paso_h=2
		elif 8<abs(min_lon-max_lon)<=30:
			paso_h=5
		elif 30<abs(min_lon-max_lon)<=100:
			paso_h=10
		else:
			paso_h=15
	else:
		paso_h=dlon
	#
	
	if bg!="None":
		if bg=="BM":
			if res=="low" or res=="medium" or res=="high" or res=="full":
				mapa.background_img(name=bg, resolution=res)
			else:
				raise ValueError('aviable resolutions to BM background are low, medium, high, full')
		elif bg=="product":
			if res=="low" or res=="high" :
				mapa.background_img(name=bg, resolution=res)
			else:
				raise ValueError('aviable resolutions to vegetation background are low, high')
		elif bg=="topo":
			if res=="low" or res=="high" :
				mapa.background_img(name=bg, resolution=res)
			else:
				raise ValueError('aviable resolutions to topography background are low, high')
		elif bg=="stock":
			mapa.stock_img()
		elif bg=="define_color":
			mapa.add_feature(cfeature.LAND,color=landcolor) #If I comment this => all ok, but I need 
			mapa.add_feature(cfeature.OCEAN,color=oceancolor)
		else:
			raise ValueError('aviable backgrounds are BM, topo, product')
	
	
	gl = mapa.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=0.5, color='black', alpha=1, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right = False
	gl.xlines = True

	lons=np.arange(floor(min_lon-paso_h),ceil(max_lon+paso_h),paso_h)
	gl.xlocator = mticker.FixedLocator(lons)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': fontsize, 'color': 'black'}
	gl.ylabel_style = {'size': fontsize,'color': 'black'}
	
	
	return mapa



def get_map_all(lower_left_corner=(255,19),upper_right_corner=(290,30),dlon=None,dlat=None,bg="None",
				res="medium",cr="10m",landcolor="#bfbfbf",oceancolor="#b8dffe",fontsize=15,id_=111,center=180):
	"""
	To create a map with Cartopy
	
	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
		
	Parameters
	-------------
	
	
	lower_left_corner: coordinate like (lon,lat)
	
	upper_right_corner: coordinate like (lon,lat)
	
	bg: string to define map background_img
	
	aviable options
	None: to use  empty map
	BM: to use Bluemarble background, aviable resolutions: low, medium, high, full
	product : to use vegetation background, aviable resolutions:low, high
	topo: to use topography background, aviable resolutions: low, high
	stock: default cartopy background, use default resolution
	define_color:to set specific colors to land and ocean define by landcolor and oceancolor 
	
	res: string to get background resolution
	cr: string Coast resolution
	aviable options: 110m or 10m
	
	fontsize:float
	to set fontsize to plot draw_labels
	
	id_: float
	position in figure (111 is default to plot in all figure)
	Return a created map


	"""
	
	
	min_lon,min_lat=lower_left_corner
	max_lon,max_lat=upper_right_corner
	
	crs = ccrs.PlateCarree()
	mapa=plt.subplot(id_,projection=ccrs.PlateCarree(center))
	
	if cr=="110m" or cr=="10m":
		mapa.add_feature(cfeature.COASTLINE.with_scale(cr), linewidth=1)
	else:
		raise ValueError('Aviable coast resolution are "110m" and "10m"')
	mapa.add_feature(cfeature.STATES, linewidth=0.25)
	mapa.set_extent([min_lon,max_lon,min_lat,max_lat], crs=ccrs.PlateCarree())
	
	if dlon==None:
		if abs(min_lon-max_lon)<=2:
			paso_h=0.5
		elif 3<abs(min_lon-max_lon)<=8:
			paso_h=2
		elif 8<abs(min_lon-max_lon)<=30:
			paso_h=5
		elif 30<abs(min_lon-max_lon)<=100:
			paso_h=10
		else:
			paso_h=15
	else:
		paso_h=dlon
	#
	if dlat==None:
		dlat=5
	else:
		dlat=dlat
	
	if bg!="None":
		if bg=="BM":
			if res=="low" or res=="medium" or res=="high" or res=="full":
				mapa.background_img(name=bg, resolution=res)
			else:
				raise ValueError('aviable resolutions to BM background are low, medium, high, full')
		elif bg=="product":
			if res=="low" or res=="high" :
				mapa.background_img(name=bg, resolution=res)
			else:
				raise ValueError('aviable resolutions to vegetation background are low, high')
		elif bg=="topo":
			if res=="low" or res=="high" :
				mapa.background_img(name=bg, resolution=res)
			else:
				raise ValueError('aviable resolutions to topography background are low, high')
		elif bg=="stock":
			mapa.stock_img()
		elif bg=="define_color":
			mapa.add_feature(cfeature.LAND,color=landcolor) #If I comment this => all ok, but I need 
			mapa.add_feature(cfeature.OCEAN,color=oceancolor)
		else:
			raise ValueError('aviable backgrounds are BM, topo, product')
	
	
	gl = mapa.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=0.5, color='black', alpha=1, linestyle='--')

	
	lons=np.arange(min_lon,max_lon,paso_h)
	
	gl_lon_info=[]
	for clons in lons:
		if clons<180:
			gl_lon_info=np.append(gl_lon_info,clons)
		else:
			gl_lon_info=np.append(gl_lon_info,clons-360)
	
	
	#gl_lon_info=[160,180,-20,-40,-60,-80,-100,-120,-140,-160,-180,-200,-220,-240,-260]
	gl_loc=[True,False,False,True]
	gl.ylabels_left = gl_loc[0]
	gl.ylabels_right = gl_loc[1]
	gl.xlabels_top = gl_loc[2]
	gl.xlabels_bottom = gl_loc[3]

	gl.xlocator = mticker.FixedLocator(gl_lon_info)
	gl.ylocator = mticker.MultipleLocator(dlat)
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.xlabel_style = {'size': fontsize, 'color': 'k'}
	gl.ylabel_style = {'size': fontsize, 'color': 'k'}
	
	
	return mapa,crs
