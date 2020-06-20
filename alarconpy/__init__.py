# Copyright (c) 2018 Albenis Perez Alarcon.
# Distributed under free License.
# apalarcon1991@gmail.com
"""Tools for reading, calculating, and plotting with weather data."""

# What do we want to pull into the top-level namespace?

import logging
import warnings

from .footer import *
from .calc import *
from .plots import *
from .plots_hurricanes import *
from .paths import *
from .cb import *
from .point_interpolation import  points_interpolation
from .mass_consistente import UVNonDivergent 
from .create_map import get_map, get_map_all
from .convert_units import *
from .statistics import *
from ._citing import get_cite
from .met_functions import *
# Must occur before below imports
warnings.filterwarnings('ignore', 'numpy.dtype size changed')

from ._version import get_versions  # noqa: E402

__version__ = get_versions()['version']
__author__ = get_versions()['author']
__contact__ = get_versions()['contact']
__last_update__ = get_versions()['last_update']
del get_versions

try:
    # Added in Python 3.2, will log anything warning or higher to stderr
    logging.lastResort
except AttributeError:
    # Add our own for MetPy on Python 2.7
    logging.getLogger(__name__).addHandler(logging.StreamHandler())


