import numpy as np
import matplotlib.pylab as plt
import sys

def vperfil_holland(vmax,rmax,r):
	"""
	Perfil de Holland (1980)
	"""

	b=2
	net=np.exp(1-(r/rmax)**(-b))
	v=(((rmax/r)**b)*net)**(0.5)
	v=vmax*v
	return v



###################################################################
#Emanuel2011
#############################################################
def vperfil_emanuel2011(r,rmax,vmax,f=1e-5):
	"""
	Perfil de Emanuel and Rotuno (2011)
	"""
	v=2*r*(rmax*vmax+0.5*f*rmax**2)/(rmax**2+r**2)-np.abs(f)*r/2.
	return v



###################################################################
#Di maria 1987
#############################################################
def vperfil_Dimaria(r,rmax,vmax,c=0.63,d=1):
	"""
	Perfil de DeMaria (1987)
	"""
	v=vmax*(r/rmax)*np.exp(((1/c)*(1-(r/rmax)**c))/d)
	return v




##################################################################
# Willougbhy three way
##################################################################
def w(e):
   return 126.0*e**5-420.0*e**6+540.0*e**7-315.0*e**8+70.0*e**9

def E(r, r1, r2): 
   return (r-r1)/(r2-r1)

def vwill_new(r, r1, r2, vmax, rm,lat):
   """
   Perfil de Willougbhy et al (2006)
   """
   
   
   x2=45
 # calculo de los coeficientes para un vmax y rm dados
   x1= 287.6 - 1.942*vmax + 7.799*np.log(rm)+1.819*np.abs(lat)
   n=2.1340 + 0.0077*vmax - 0.4522*np.log(rm)-0.0038*np.abs(lat)
   A = 0.5913 + 0.0029*vmax-0.1361*np.log(rm)-0.0042*np.abs(lat)
   
   if r <= r1: 
    return vmax*(r/rm)**n
   Vi = vmax*(r1/rm)**n
   
   #print vmax*(rm/rm)**n
   V0 = vmax*((1-A)*np.exp(-(r2-rm)/x1)+A*np.exp(-(r2-rm)/x2))
   #print w(E(rm, r1, r2))
   #print Vi*(1-w(E(rm, r1, r2))) + V0*w(E(rm, r1, r2))
   #print vmax*((1-A)*np.exp(-(rm-rm)/x1)+A*np.exp(-(rm-rm)/x2))
   if r1 < r <= r2:   
    return Vi*(1-w(E(r, r1, r2))) + V0*w(E(r, r1, r2))
   return vmax*((1-A)*np.exp(-(r-rm)/x1)+A*np.exp(-(r-rm)/x2))





#######################################################
#Frisis-Scogeman(2013)
########################################################

def vfrisius(r,rmax,vmax,fcor="1e-5",a=1):
	"""
	Perfil de Fisuis et al (2013)
	"""

	Mm=rmax*vmax

	#v=2*vmax*rmax/(r*(1+(rmax/r)**2))-0.5*fcor*r
	deno=(2.-a+a*((r/rmax)**2.))
	nume=(2.*((r/rmax)**2.))
	v=(Mm/r)*(nume/deno)**(1./(2.-a))-0.5*np.abs(fcor)*r
	return v



#######################################################
# Holland2010
######################################################


def vperfil_holland_2010(vmax,rmax,r,bs=1.8):
	"""
	Perfil de Holland et al. (2010)
	"""
	x1=0.5-((r-rmax)/(90*rmax))
	#x1=-r/(90*rmax)+46/90.
	bs=1.8
	net=np.exp(1-(r/rmax)**(-bs))
	v=(((rmax/r)**bs)*net)**(x1)
	v=v*vmax
	return v

###################################################
#Emanuel 2004
#################################################
def vperfil_emanuel(r,rmax,vmax,ro,n=0.9,m=1.6,a=0.25):
	"""
	Perfil de Emanuel (2004)
	"""

	aux1=(a*(1+2*m)/((1+2*m*((r/rmax)**(2*m+1)))))

	aux2=((1-a)*(n+m)/((n+m*((r/rmax)**(2*(n+m))))))
	#**
	#**************
	aux3=(((ro-r)/(ro-rmax))**2)*((r/rmax)**(2*m))
	#*****************

	v=np.sqrt((vmax**2)*aux3*(aux2+aux1))
	return v
