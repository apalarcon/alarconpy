import numpy as np
import metpy.calc as cl
from datetime import *
from scipy.interpolate import interp1d
from datetime import datetime, date,timedelta
import math


def hr_calc(T,p,q):
	"""Calculate relative humidity
	
	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
		
	Parameters
	----------
	
	T : float or array 2D
	single value or array of temperature (en grados celsius C = K - 273.15)

	P : float or numpy 2d array
	single value or array of surface pressure in hPa

	q : single value or numpy 2d array
	single value or array of specific humidity

	
	Returns
	-------
	single value or array of relative humidity

	"""

	es = 6.112 * np.exp(17.67 * T/(T + 243.5))
	w = q/(1-q)
	e = (w * p / (.622 + w))
	dew_T = (243.5 * np.log(e/6.112))/(17.67-np.log(e/6.112))


	ens=6.1*10*(7.5*dew_T/(237.7+dew_T))
	esa=6.1*10*(7.5*T/(237.7+T))
	hr = 100. * (ens/esa)

	return hr



def time_dif(initdate,enddate):
	"""
	To calculate de diference between two dates

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	
	initdate : int
	init day in format YYYYMMDDHHmmss

	enddate : int
	end day in format YYYYMMDDHHmmss


	return the diference between enddate and initdate

	"""
	year_init=int(str(initdate)[0:4])
	mes_init=(int(str(initdate)[4:6]))
	day_init=(int(str(initdate)[6:8]))
	hour_init=(int(str(initdate)[8:10]))
	min_init=(int(str(initdate)[10:12]))
	seg_init=(int(str(initdate)[12:14]))
	
	year_end=int(str(enddate)[0:4])
	mes_end=(int(str(enddate)[4:6]))
	day_end=(int(str(enddate)[6:8]))
	hour_end=(int(str(enddate)[8:10]))
	min_end=(int(str(enddate)[10:12]))
	seg_end=(int(str(enddate)[12:14]))
	


	fecha1 = datetime(year_init, mes_init, day_init, hour_init, min_init, seg_init)
	fecha2 = datetime(year_end, mes_end, day_end, hour_end, min_end, seg_end)
	diferencia = fecha2 - fecha1

	return int((diferencia.total_seconds())/3600.)



def split_string(field,separador):
	"""
	To separate string

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	
	field : str
	

	separador : str
	


	return a list of string

	"""


	string=[]
	aux = field.split(separador)
	#string=np.append(string,aux[0])
	return aux


def index_row(myList, v):
	"""
	To find string in a list
	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	
	myList : str
	cadena de caracteres

	v : str
	caracter a buscar


	return index of strig found

	"""


	j=[]
	for i, x in enumerate(myList):
     
		if v in x:
			j.append(i)
	return j



def wind_rose_dir(dir_degree):
	"""
	To convert wind direction en degree to direction of nautical rose

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	
	dir_degree : float
	wind direction in degree

	return wind direction in nautical rose

	"""


	if dir_degree >= 0 and dir_degree<11.25:
		dd='N'
	if dir_degree >= 11.25 and dir_degree<33.75:
		dd='NNE'
	if dir_degree >= 33.75 and dir_degree<56.25:
		dd='NE'
	if dir_degree >= 56.25 and dir_degree<78.75:
		dd='ENE'
	if dir_degree >= 78.25 and dir_degree<101.25:
		dd='E'
	if dir_degree >= 101.25 and dir_degree<123.75:
		dd='ESE'
	if dir_degree >= 123.75 and dir_degree<146.25:
		dd='SE'
	if dir_degree >= 146.25 and dir_degree<168.75:
		dd='SSE'
	if dir_degree >= 168.75 and dir_degree<191.25:
		dd='S'
	if dir_degree >= 191.25 and dir_degree<213.75:
		dd='WSW'
	if dir_degree >= 213.75 and dir_degree<236.25:
		dd='SW'
	if dir_degree >= 236.25 and dir_degree<258.75:
		dd='WSW'
	if dir_degree >= 258.75 and dir_degree<281.25:
		dd='W'
	if dir_degree >= 281.25 and dir_degree<303.75:
		dd='WNW'
	if dir_degree >= 303.75 and dir_degree<326.25:
		dd='NW'
	if dir_degree >= 326.25 and dir_degree<348.75:
		dd='NNW'
	if dir_degree >= 348 and dir_degree <360:
		dd='N'

	return dd



def storm_type(wind):
	"""
	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	wind: flota
	max wind speed in km/h
	return a type of tropical cyclone
	"""


	if wind<32:
		type_storm="Observación"
	if wind>=32 and wind<62:
		type_storm="Depresión Tropical"
	if wind>=62 and wind<119:
		type_storm="Tormenta Tropical"
	if wind>=119:
		type_storm="Huracán"
	return type_storm



def t_from_tp(T,p):
	"""
	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	tp: float
	array of temperatures in Kelvin
	P: float
	array of pressure
	return a potntial temperature
	"""
	return T * (1000.0/p) ** (2./7.)



def interp_levels(x, y, levels):
    """
    Author: Albenis Pérez Alarcón
    contact:apalarcon1991@gmail.com
	
    Parameters
    ----------
    To interpolate to pressure levels
    x: array pressure
    y:array of variable
    """
    
    shp = y.shape
   

    x = np.reshape(x, [shp[0],-1]).T
    y = np.reshape(y, [shp[0],-1]).T
    
    
    #print (x.shape,y.shape)
    
    #sys.exit()
    new_shape = (len(x), len(levels))

    values = np.empty( new_shape )
    for i, (xo, yo) in enumerate(zip(x, y)):
        
        interp_mod = interp1d(xo, yo, fill_value='extrapolate', axis=0)
        values[i] = interp_mod(levels)
        
    return values.T.reshape( new_shape[1:] + shp[1:] )



def gdi_index(t950, t850, t700, t500, r_950, r_850, r_700, r_500,psfc):
	"""
	Author: Albenis Pérez Alarcón  in colaboration with José Carlos Fernández Alvarez and Tahimy Fuentes
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	tp: float
	array of temperatures in Kelvin in diferente levels, subindices explain that levels
	r: float
	array of mixing ratio in diferents levels, subindices explain that levels
	return a gdi index values
	"""

	tp_950=t_from_tp(t950,950.) 
	tp_850=t_from_tp(t850,850.)
	tp_700=t_from_tp(t700,700.)
	tp_500=t_from_tp(t500,500.)
	
	cpd = 1005.7
	l0 = 2.69e6
	
	#Layer A
	tpa = tp_950
	r_a = r_950

	#Layer B
	tpb = 0.5*(tp_850 + tp_700)
	r_b = 0.5*(r_850 + r_700)

	#Layer C
	tpc = tp_500
	r_c = r_500

	#EPT
	eptp_a = tpa*np.exp((l0*r_a)/(cpd*t850))
	eptp_b = tpb*np.exp((l0*r_b)/(cpd*t850)) - 10.
	eptp_c = tpc*np.exp((l0*r_c)/(cpd*t850)) - 10.

	#Column Buoyancy Index 
	me = eptp_c - 303.
	le = eptp_a - 303.

	cbi = np.where( le>0.0, 6.5e-2 * le*me, 0.0 )

	#Mid tropospheric Warming/Stabilization Index 
	mwi = np.where( t500>263.15, -7.0 * (t500-263.15), 0.0 )

	#Inversion Index
	sd = 1.5 * ( (t950 - t700) + (eptp_b - eptp_a) )

	ii = np.where( sd>0.0, 0.0, sd )

	#Terrain correction 
	p1 = 500.
	p2 = 9000.
	p3 = 18.

	tc = p3 - p2/(0.01*psfc - p1)
	#print ("correction:",tc)

	return cbi + mwi + ii+tc

def time_to_UTC(mytime,utc_offset):
	"""
	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	mytime:string
	date in format yyyy-mm-dd hh:mm:ss

	utc_offset: int
	hours differents from Greenwich Meridian

	return time in UTC format
	"""

	local_time = datetime.strptime(mytime, "%Y-%m-%d %H:%M:%S")
	utc_time=local_time-timedelta(hours=utc_offset)

	return utc_time


def time_calc(init_time,h_diff):
	"""
	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------

	init_time:string
	date in format yyyy-mm-dd hh:mm:ss

	h_diff: int
	hours since init_time

	return time 
	"""

	formatted_time = datetime.strptime(init_time, "%Y-%m-%d %H:%M:%S")
	calculated_time=formatted_time+timedelta(hours=h_diff)

	return calculated_time

def haversine(origin, destination,units='km',ellipsoid="WGS84"):
	"""
	To calculate distance between two points in Earth giving their coordinates (lat,lon)

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------
	origin:array like (lat,lon)
	coordinates of origin point

	destination: array like (lat,lon)
	coordinates of destinations points

	units: str
	units to return distance
	aviable units are kilometers (km), meters (m) and miles (mi)
	
	ellipsoid: String: type of projection
	aviables: Airy (1830), Bessel,Clarke (1880),FAI sphere,GRS-67,International,Krasovsky,NAD27,WGS66,WGS72,IERS (2003),WGS84- default WGS84
	return
	distance between points
	"""
	if units=="km" or units=="kilometers":
		factor=1
	elif units=="m" or  units=="meters":
		factor=1000
	elif units=="miles" or units=="mi":
		factor=0.621371
	else:
		raise ValueError('aviable units are kilometers (km), meters (m) and miles (mi)')


	

	lat0, lon0 = origin
	
	
	ellipsoids = {
        "Airy (1830)": (6377.563, 6356.257), # Ordnance Survey default
        "Bessel": (6377.397, 6356.079),
        "Clarke (1880)": (6378.249145, 6356.51486955),
        "FAI sphere": (6371, 6371), # Idealised
        "GRS-67": (6378.160, 6356.775),
        "International": (6378.388, 6356.912),
        "Krasovsky": (6378.245, 6356.863),
        "NAD27": (6378.206, 6356.584),
        "WGS66": (6378.145, 6356.758),
        "WGS72": (6378.135, 6356.751),
        "WGS84": (6378.1370, 6356.7523), # GPS default
        "IERS (2003)": (6378.1366, 6356.7519),
    }
	
	r1, r2 = ellipsoids[ellipsoid]
    
	
	
	
	
	
	lat, lon = destination
	
	mean_latitude = (lat0 + lat)/2
	A = (r1*r1 * math.cos(mean_latitude))**2
	B = (r2*r2 * math.sin(mean_latitude))**2
	C = (r1 * math.cos(mean_latitude))**2
	D = (r2 * math.sin(mean_latitude))**2
	radius  = np.sqrt(( A + B )/( C + D)) #radius of the earth in km
	
	dlat = math.radians(lat-lat0)
	dlon = math.radians(lon-lon0)
	a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat0)) \
	* math.cos(math.radians(lat)) * math.sin(dlon/2) * math.sin(dlon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	distance=radius * c*factor

	return distance

def find_lim(a, b, delta = 0, xcenter=-82.11, ycenter=23.37):
    __doc____ = 'esta funcion devuelve los limites del menor subdominio del meshgrid de b[0][:] y b[1][:] siendo estos ' \
                'arreglos que contenga el meshgrid de a[0][:] y b[1][:] siendo estos dos arreglos'

    import numpy as np
    k=6
    it=np.argmin(np.sqrt((a-xcenter)**2+(b-ycenter)**2))
    
    i,j = np.int(it/np.size(a[0])), np.mod(it,np.size(a[0]))
    lim = [[i-k, i+k],[ j-k+3, j+k-2]]
    
    return lim


def haversine_vectorized(lat0, lon0,lat1,lon1, units="km",ellipsoid="WGS84"):
	"""
	To calculate distance between points in two arrays

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com
	
	Parameters
	----------------------------
	lat0 and lat1: array with latitudes like (N,)
	lon0 and lon1:array with longitudes  (N,)
	
	units: str
	units to return distance
	aviable units are kilometers (km), meters (m) and miles (mi)
	
	ellipsoid: String: type of projection
	aviables: Airy (1830), Bessel,Clarke (1880),FAI sphere,GRS-67,International,Krasovsky,NAD27,WGS66,WGS72,IERS (2003),WGS84- default WGS84
	
	
	return array with distance between points 
	
	"""
	rm = []
	for i in range(len(lat0)):
		rm=np.append(rm,haversine((lat0[i], lon0[i]), (lat1[i], lon1[i]), units,ellipsoid))
	return np.array(rm)


def convert_pressure_to_height(p_upper):
	"""
	Convert pressure levels to height in feet

	Author: Albenis Pérez Alarcón
	contact:apalarcon1991@gmail.com

	Parameters 
	--------------------------
	p_upper : float
	level pressure in hPa

	the expression used to calculate altitude was taken from https://www.weather.gov/media/epz/wxcalc/pressureAltitude.pdf

	to convert to meter used hm=0.3448*z
	"""

	#z=(10**((np.log10(p_upper/mslp))/(2.2558797) ) -1)/(-6.8755856E-6)

	z= (1- (p_upper/1013.25)**0.190284)*145366.45

	return z


def vertical_cut(lat_storm=None,lon_storm=None,lat=np.array([None]),lon=np.array([None]),var=np.array([None]),cut_type="zonal",points_interval=20):
	"""
	Author: Albenis Pérez Alarcón
	
	This function make a verical cut for tridimensional variables
	
	lat_storm: float
	It is the latitude of the point where it will  vertical cut
	
	lon_storm: float
	It is the longitude of that point of the point where it will  vertical cut
	
	lat: numpy array (N,D)
	Contains the latitude of all point of grid
	
	lon=numpy array (N,D)
	Contain the longitude of all points of grid
	
	var: tuple array (H,N,D)
	contains the values of the variable in tridimensional array
	
	cut_type: string
	It is if the vertical_cut will meridional o zonal
	
	points_interval: int
	
	Number of points that it will taken to do a zonal or meridional verical cut
	
	--------------
	Return 
	
	new_var: numpy array (H,N) or (H,D)
	Contains the values of variable after vertical cut
	
	new_cor: numpy array (2*points_interval) 
	Contain the latitude or longitude of all point in new_var
	"""
	
	
	if lat_storm==None or lon_storm==None:
		raise ValueError("You must provide a lat_storm and lon_storm position")
	elif lat.any()==None or lon.any()==None:
		raise ValueError("You must arrays with lat and lon")
	elif var.any()==None:
		raise ValueError("You must provide a variable to do vertical cut")
	
	
	dist=np.empty_like(var[0,:])
	
	for i in range(0,dist.shape[0]):
		for j in range(0,dist.shape[-1]):
			dist[i,j]=haversine((lat_storm,lon_storm),(lat[i,j],lon[i,j]))
	
	
	i_storm,j_storm=np.where(dist==dist.min())
	i_storm=i_storm[0]
	j_storm=j_storm[0]
	
	
	if cut_type=="zonal" and j_storm+points_interval> var.shape[2] or j_storm-points_interval<=0:
		raise ValueError("You must provide a valid points_interval. Check that the provided points_interval value + j_storm position is not greather than var dimension or j_storm - points_interval is less than zero")
	elif cut_type=="meridional" and i_storm+points_interval> var.shape[1] or i_storm-points_interval<=0:
		raise ValueError("You must provide a valid points_interval. Check that the provided points_interval value + i_storm position is not greather than var dimension or i_storm - points_interval is less than zero")
	
	
	if cut_type=="zonal":
		new_var=var[:,i_storm,j_storm-points_interval:j_storm+points_interval]
		new_cor=lon[i_storm,j_storm-points_interval:j_storm+points_interval]
	elif cut_type=="meridional":
		new_var=var[:,i_storm-points_interval:i_storm+points_interval,j_storm]
		new_cor=lat[i_storm-points_interval:i_storm+points_interval,j_storm]
	else:
	
		raise ValueError("cut_type must be 'zonal' or 'meridional'")
	
	return new_var,new_cor

def geo_submatrix(lat,lon,var,lat_min,lat_max,lon_min,lon_max):
	"""
	Author: Albenis Pérez Alarcón
	
	This function make a submatrix
	
	lat: numpy array (N,M)
	array of latitudes
	
	lon: numpy array (N,M)
	array of longitudes
	
	var: numpy array (N,M)
	array to extract submatrix
	
	lat_min and lat_max: float
	min and max latitude to make the submatrix
	
	lon_min and lon_max: float
	min and max longitude to make the submatrix
	
		
	--------------
	Return 
	
	nvar: variable submatrix
	nlat and nlon: geographic coordinates submatrix
	"""
	
	l1=(lat>=lat_min)&(lat<=lat_max)
	l2=(lon>=lon_min)&(lon<=lon_max)

	m=l1&l2

	r,c=0,0

	for l in range(m.shape[0]):
		if m[l,:].sum():
			c=m[l,:].sum()
			break
	for l in range(m.shape[1]):
		if m[:,l].sum():
			r=m[:,l].sum()
			break

	nlon=lon[l1&l2].reshape(r,c)
	nlat=lat[l1&l2].reshape(r,c)
	nvar=nvar[l1&l2].reshape(r,c)
	
	return nlat,nlon,nvar
