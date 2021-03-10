#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Download ERA5 data the Copernicus Climate Change Service API (https://cds.climate.copernicus.eu/)

More examples here: https://github.com/esowc/ml_flood/tree/master/notebooks/1_data_download_analysis_visualization

"""

# Get your UID and API key from https://cds.climate.copernicus.eu/user and insert it in the variables below
UID = '20018'
API_key = '178524fd-b8ed-4702-be62-ad1797c7a002'

# Write the keys into the file ~/.cdsapirc in the home directory of your user
import os
with open(os.path.join(os.path.expanduser('~'), '.cdsapirc2'), 'w') as f:
    f.write('url: https://cds.climate.copernicus.eu/api/v2\n')
    f.write(f'key: {UID}:{API_key}')
    
# Start a request
# You will be asked to agree to the terms of use from the copernicus climate data store for your first download.

# Import cdsapi and create a Client instance
import cdsapi
c = cdsapi.Client()

# More complex request
c.retrieve("reanalysis-era5-pressure-levels", {
            "product_type":   "reanalysis",
            "format":         "netcdf",
            "area":           "52.00/2.00/40.00/20.00", # N/W/S/E
            "variable":       "geopotential",
            "pressure_level": "500",
            "year":           "2017",
            "month":          "01",
            "day":            "12",
            "time":           "00"
            }, "example_era5_geopot_700.nc")