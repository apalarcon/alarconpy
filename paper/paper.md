---
title: 'Alarconpy: a Python Package for Meteorologists'
tags:
  - Python
  - meteorology
  - programming  languages and software
  - atmospheric sciences
  - scientific programming
  - meteorological variables
  - weather data processing
  - python packages
  - development software
authors:
  - name: Albenis Pérez-Alarcón^*
    orcid: 0000-0002-9454-2331
    affiliation: "1" 
  - name: José Carlos Fernández-Alvarez
	orcid: 0000-0003-3409-6138
    affiliation: "1"
affiliations:
 - name: Department  of Meteorology, Higher Institute of Technologies and Applied Sciences, University of Havana
   index: 1

date: 19 June 2020
bibliography: paper.bib

# Summary
In atmospheric science, researchers generally must process a large volume of data obtained from the network of meteorological observations on a local and global scale, as well as from the output of numerical weather forecast models. This data collection is stored in netCDF (Network Common Data Form) and GRIB (GRIdded Binary or General Regularly-distributed Information in Binary form, @WMO2003) format. GRIB is a concise data format commonly used in meteorology to store forecast and historical meteorological data. It is standardized by the Commission for Basic Systems of the World Meteorological Organization (WMO).

These formats for storing meteorological data have led to the development of several applications for their processing, such as the NetCDF Operators (NCO) and the Climate Data Operators (CDO). The former focuses on simple data curation (e.g. viewing the contents of a file, selecting a subset of the data or editing the metadata within a file), while the latter provides for simple statistical analysis (e.g. calculating the climatology, percentile, correlation or heat wave index). It has also been observed in a rapid development of libraries in different programming languages such a Fortran, C, C++, MATLAB (MATrix LABoratory), Python, NCL (National Center for Atmospheric Research (NCAR) Command Language) and IDL (Interactive Data Language) to manage them.

Python is a modern, open-source, interpreted computer language whose use in the atmospheric and oceanic sciences is growing by leaps and bounds [@Irving2019].  In addition, Python offers the IPython interactive command line that allows scientists to view data, test new ideas, combine algorithmic approaches, and evaluate their results directly. This process could lead to an end result, or it could clarify how to build broader and more static production code [ @Perez2007 ]. Therefore, Python is an excellent tool for such a workflow [ @Yan1998 ]. In fact, the development in Python of several libraries such as Metpy [@metpy] and Siphon [@siphon] have made it possible to integrate the functionalities of various standard Python packages in the processing of weather and oceanic data.

The flexibility of Python as a programming language as well as the functions provided by various Python packages for the processing of weather data, motivated the development of the Alarconpy package. It integrates facilities that separately provide  libraries such as Metpy [@metpy], Scipy ({https://www.scipy.org}) and  Cartopy ({https://scitools.org.uk/cartopy/docs/latest/}), as well as the development of new applications. In this way, the user is provided with a set of built-in tools for rapid data processing.

Alarconpy is a set of tools developed in Python  for the processing of meteorological data. It  currently  version is 1.0.3  and it includes functions to calculate  some atmospheric variables such as relative humidity and the saturation vapor pressure, operations with dates and a module with specific algorithms for working with tropical cyclones (TCs). It also includes several predefined color palettes for plotting the wind field, the surface and mean sea level pressure, the radar reflectivity, the Gálvez-Davison Index [@Galvez2016],  the precipitation as well as the color palettes available in Basemap ({https://basemaptutorial.readthedocs.io/en/latest/}). 


# Acknowledgements

Many thanks to Roberto Carlos Cruz Rodrı́guez for all knowledge transmitted on the use of the Python
programming language.


# References
