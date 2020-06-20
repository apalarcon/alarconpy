import numpy as np
from alarconpy.calc import haversine
from netCDF4 import Dataset
from numpy import dtype


def create_TCmask(latc,lonc,radius,dlat=20,dlon=25,res=0.5):
	"""
	Albenis Pérez Alarcón
	Create circular TC mask
	
	latc: float
	latitude of the TC center (degrees)

	lonc: float
	longitude of the TC center (degrees)

	radius: float

	TC outer radius (km)


	dlat and dlon: int
	to create domain

	res: float
	horizontal resolution

	return
	array of lat, lon and mask

	"""

	lat=np.arange(latc-dlat,latc+dlat,res)
	lon=np.arange(lonc-dlon,lonc+dlon,res)
	
		
	for i in range(0,len(lon)):
		if lon[i]<-180:
			lon[i]=lon[i]+360

	lonn,latt=np.meshgrid(lon,lat)

	mask=np.empty_like(lonn)
	mask[:]=0

	for i in range(0,latt.shape[0]):
		for j in range(0,latt.shape[1]):
			mask[i,j]=haversine((latc,lonc),(latt[i,j],lonn[i,j]))

	
	for i in range(0,len(lon)):
		if lon[i]<-0:
			lon[i]=lon[i]+360
				
			
	mask[mask<=radius]=1
	mask[mask>radius]=0
	

	
	return lat,lon, mask



def TCmask_nc(latitude,longitude,var,filename="output"):
	"""
	Albenis Pérez Alarcón
	to creat a netcdf file with TC mask
	
	latitude: numpy array (N,)
	longitude: numpy array (M,)
	
	var:numpy array (N,M)
	array of mask file
	
	output: string
	path and file name netcdf file (without nc extension)
	
	return
	return a netcdf file
	
	"""
	
	
	ncout = Dataset(filename+".nc", 'w', format='NETCDF4')
	# define axis size
	ncout.createDimension('lat', len(latitude))
	ncout.createDimension('lon', len(longitude))

	# create latitude axis
	lat = ncout.createVariable('lat', dtype('float').char, ('lat'))
	lat.standard_name = 'latitude'
	lat.long_name = 'latitude'
	lat.units = 'degrees'
	lat.axis = 'Y'

	# create longitude axis
	lon = ncout.createVariable('lon', dtype('float').char, ('lon'))
	lon.standard_name = 'longitude'
	lon.long_name = 'longitude (0-360)'
	lon.units = 'degrees'
	lon.axis = 'X'

	# create variable array
	vout = ncout.createVariable('mask', dtype('float').char, ('lat','lon'))
	vout.long_name = 'Mask'
	vout.units = ''
	vout.standard_name = "TC mask" ;
	vout.coordinates = "lat,lon" ;
	#vout._FillValue = 1.e20 ;
	vout.original_name = "mask"

	lon[:] = longitude[:]
	lat[:]= latitude[:]
	vout[:] = var[:]
	ncout.close()



def basin_code(basin):
	"""
	Albenis Pérez Alarcón
	Function to determine TC basin basin_code
	
	basin: str
	Basin name
	NATL: North Atlantic
	NEPAC: Central and East Pacific
	WNP: Western North Pacific
	NIO: North Indian Ocean
	SIO: South Indian Ocean
	SPO: South Pacific Ocean
	
	return
	basin code
	"""
	
	
	if basin=="NATL":
		bcode="AL"
	elif basin=="NEPAC":
		bcode="EP"
	elif basin=="WNP":
		bcode="WP"
	elif basin=="NIO":
		bcode="IO"
	elif basin=="SIO":
		bcode="SI"
	elif basin=="SPO":
		bcode="SP"
	
	return bcode



def polar_cords(radius,latc,lonc):
	"""
	Albenis Pérez Alarcón
	
	Function to create a circle around the TC center
	
	radius: float
	TC outer radius
	
	latc: float
	TC latitude center (degress)
	
	lonc: float
	TC longitude center (degress)
	
	return
	lat, lon 1D arrays
	"""
	
	
	
	
	radius=radius/111

	theta=np.arange(0,2*np.pi,np.pi/32)

	lat=[]
	lon=[]
	for j in range(0,len(theta)):
		lon=np.append(lon, radius*np.cos(theta[j]))
		lat=np.append(lat, radius*np.sin(theta[j]))
		



	for j in range(0,len(theta)):
		lon[j]=lon[j]+lonc
		lat[j]=lat[j]+latc


	lon=np.append(lon,lon[0])
	lat=np.append(lat,lat[0])
	

	return lat,lon




def calc_eminuspArea(latres,lonres):
	"""
	Albenis Péres Alarcón
	Function to calc eminusp Area
	
	latres: float
	resolution of array of latitudes
	
	lonres:
	resolution of array of longitudes
	
	return
	
	Area of each grid
	"""
	
	rt = 6371000. #Radio medio de la Tierra en metros
	gr = np.pi/180.
	
	
	latitude=np.arange(0,181,latres)
	
	lat=[]
	for mlat in latitude:
		lat=np.append(lat,90-mlat)
		
	lon=np.arange(0,360,lonres)	

	area=np.empty((len(lat)-1,len(lon)))
	area[:,:]=0
	for i in range(0,len(lat)-1):
		for j in range(0,len(lon)):
			area[i,j]=np.abs((gr*rt**2)*( np.sin(gr*lat[i]) - np.sin(gr*lat[i+1])))*np.abs(lonres)
			
			
	#print(area.max(),area.min())
	return area


calc_eminuspArea(1,1)
