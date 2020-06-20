# -*- coding: utf-8 -*-
"""
Demo of unicode support in text and labels.
"""

from __future__ import unicode_literals
import numpy as np
from scipy.stats import linregress
from scipy import stats
import matplotlib.pylab as plt
import sys


def filter_nan(s,o):
    """
    this functions removed the data  from simulated and observed data
    whereever the observed data contains nan
    
    
    
    this is used by all other functions, otherwise they will produce nan as 
    output
    """
    data = np.array([s.flatten(),o.flatten()])
    data = np.transpose(data)
    data = data[~np.isnan(data).any(1)]
    return data[:,0],data[:,1]


def pc_bias(s,o):
    """
    Percent Bias
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        pc_bias: percent bias
    """
    s,o = filter_nan(s,o)
    return 100.0*sum(s-o)/sum(o)

def apb(s,o):
    """
    Absolute Percent Bias
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        apb_bias: absolute percent bias
    """
    s,o = filter_nan(s,o)
    return 100.0*sum(abs(s-o))/sum(o)

def rmse(s,o):
    """
    Root Mean Squared Error
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        rmses: root mean squared error
    """
    s,o = filter_nan(s,o)
    return np.sqrt(np.mean((s-o)**2))

def mae(s,o):
    """
    Mean Absolute Error
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        maes: mean absolute error
    """
    s,o = filter_nan(s,o)
    return np.mean(abs(s-o))

def bias(s,o):
    """
    Bias
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        bias: bias
    """
    s,o = filter_nan(s,o)
    return np.mean((s-o))

def NS(s,o):
    """
    Nash Sutcliffe efficiency coefficient
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
    s,o = filter_nan(s,o)
    return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)

def L(s,o, N=5):
    """
    Likelihood 
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        L: likelihood
    """
    s,o = filter_nan(s,o)
    return np.exp(-N*sum((s-o)**2)/sum((o-np.mean(o))**2))

def correlation(s,o):
    """
    correlation coefficient
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """
    s,o = filter_nan(s,o)
    if s.size == 0:
        corr = np.NaN
    else:
        corr = np.corrcoef(o, s)[0,1]
        
    return corr


def index_agreement(s,o):
    """
	index of agreement
	
	Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
	input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    s,o = filter_nan(s,o)
    ia = 1 -(np.sum((o-s)**2))/(np.sum(
    			(np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
    return ia


def F_S(s, o):
	"""
    Forecast Skill
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        Forecast Skill
    """
	s,o = filter_nan(s,o)
	return 1-np.sqrt(np.mean((s-o)**2/np.mean(o**2)))
 
def dispersion_index(s, o):
	
	"""
    Forecast Skill
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        dispersion index
    """
	s,o = filter_nan(s,o)
	return rmse(s,o)/np.mean(o**2)

def er(s,o, N=5):
    """
    Error Relativo
    
    Adapated to use in alarconpy by Albenis Pérez Alarcón
    contact: apalarcon1991@gmail.com
    
    
    Parameters
    --------------------------
    input:
        s: simulated
        o: observed
    output:
        L: likelihood
    """
    s,o = filter_nan(s,o)
    return np.mean(abs(s-o)/o)*100
