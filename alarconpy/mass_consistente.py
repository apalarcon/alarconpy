"""
Created on Wed Nov 21 17:56:48 2018

@author: arlett
"""
import numpy as np
from scipy.interpolate import Rbf
from metpy.interpolate import interpolate_to_points


def adjust_range(a, b, dx, referencia):
    arr = np.arange(referencia, a-dx, -dx)
    arr = np.arange(arr[-1], b, dx)
    return arr


class UVNonDivergent:
    def __init__(self, xp, yp, up, vp, x_center,y_center, scale, orientacion, th, rbf_funct, bbox=None):
        self.xp = xp
        self.yp = yp
        self.up = up
        self.vp = vp
        self.th = th
        
        if bbox is None:
            self.xgrid=adjust_range( xp.min(), xp.max(), scale*np.cos(orientacion), x_center)
            self.ygrid=adjust_range( yp.min(), yp.max(), scale*np.sin(orientacion), y_center)
        else:
            self.xgrid=adjust_range( bbox[0], bbox[1], scale*np.cos(orientacion), x_center)
            self.ygrid=adjust_range( bbox[2], bbox[3], scale*np.sin(orientacion), y_center)            

        self.scale = self.xgrid[1] - self.xgrid[0]
        #if self.scale != scale:
            #print("Scale was adjusted for {} to {}".format(scale, self.scale))
                    
        self.xgrid, self.ygrid = np.meshgrid(self.xgrid, self.ygrid) 
            
        self._interpolation(rbf_funct)
        self._solve_poisson_problem()

        self.ugrid_final = self.ugrid.copy()
        self.ugrid_final[1:-1,1:-1] += self.ugrid_add

        self.vgrid_final = self.vgrid.copy()
        self.vgrid_final[1:-1,1:-1] += self.vgrid_add


    def get_interpolate_uvfield(self):
        return self.xgrid[1:-1,1:-1], self.ygrid[1:-1,1:-1], self.ugrid[1:-1,1:-1], self.vgrid[1:-1,1:-1]


    def get_non_divergent_uvfield(self):
        return self.xgrid[1:-1,1:-1], self.ygrid[1:-1,1:-1], self.ugrid_final[1:-1,1:-1], self.vgrid_final[1:-1,1:-1]


    def _interpolation(self, rbf_funct):    
        """
        Aqui se interpola los valores de viento (u,v) a los puntos de 
        rejilla usando funciones radiales. Se interpola u y v por separado.
        El tipo de interpolacion radial lo define "rbf_funct". Ver "scipy.interpolate.rbf"
        para mas informacion
        """
        #print(self.up)
        #grid_xy=None
        #self.xgrid,self.ygrid,self.ugrid= grid.interpolate(self.xp, self.yp, self.up,rbf_funct,hres=0.005,search_radius=8000)
        #self.xgrid,self.ygrid,self.vgrid= grid.interpolate(self.xp, self.yp, self.vp,rbf_funct,hres=0.005,search_radius=8000)
        mygrid=np.array(list(zip(self.xgrid.flatten(),self.ygrid.flatten())))
        mypoints=np.array(list(zip(self.xp,self.yp)))
        
        

        self.ugrid=  interpolate_to_points(mypoints, self.up, xi=mygrid,interp_type= rbf_funct,search_radius=8000)
        self.vgrid = interpolate_to_points(mypoints, self.vp,xi=mygrid,interp_type= rbf_funct,search_radius=8000)
       
        #print("******************************************",self.ugrid)
        
        #rbf_v = Rbf(self.xp, self.yp, self.vp, interp_type=rbf_funct)
            
        #self.ugrid = rbf_u( self.xgrid.flatten(), self.ygrid.flatten() )
        
       
        
       # self.vgrid = rbf_v( self.xgrid.flatten(), self.ygrid.flatten() )
        
        #print("******************************************",self.vgrid.shape)
                    
        self.ugrid.shape = self.xgrid.shape
        self.vgrid.shape = self.ygrid.shape 
        #self.xgrid=xx
        #self.ygrid=yy   
        
    def _solve_poisson_problem(self):
        """
        Aqui se resulve el problema eliptico, es decir se calcula phi 
        El sistema de ecuaciones se resuelve usando el metodo de Jacobi, sientete
        libre de modificarlo como quieras
        """
        udx = (self.ugrid[1:-1, 2:]-self.ugrid[1:-1,:-2])/(2*self.scale)
        vdy = (self.vgrid[2:, 1:-1]-self.vgrid[:-2, 1:-1])/(2*self.scale)                
        p = np.zeros(shape=self.ugrid.shape)        
        tol = 1.e-9
        er  = 1.e99
        while er > tol:
            #print(er)
            ptmp = 0.25*(p[1:-1, 2:]+p[1:-1,:-2]+p[2:, 1:-1]+p[:-2, 1:-1]) - 0.25*self.scale**2*(  -1/self.th*(udx+vdy) )              
            er = np.max( (p[1:-1,1:-1]-ptmp)**2 )             
            p[1:-1,1:-1] = ptmp[:,:]         
        self.ugrid_add = (p[1:-1,2:]-p[1:-1,:-2])/(2*self.scale)*self.th
        self.vgrid_add = (p[2:,1:-1]-p[:-2,1:-1])/(2*self.scale)*self.th

