
#%%
'''
Purpose: Accept intermediate ECICE .txt files and convert to NETCDF4 format
Author: Wayne de Jager
'''

def get_grids(hemisphere, grid_proj, grid_res):
    """
    Generate a regular polar grid (NSIDC or EASE-Grid 2.0).

    Parameters
    ----------
    hemisphere : {'north', 'south'}
    grid_proj  : {'nsidc', 'ease'}
    grid_res   : float
        Grid resolution in km.

    Returns
    -------
    lat, lon : 2D arrays
        Latitude and longitude (degrees).
    x, y : 2D arrays
        Projected x/y coordinates (meters).
    """

    resolution = grid_res * 1000.0

    hemisphere = hemisphere.lower()
    grid_proj  = grid_proj.lower()

    if grid_proj == 'nsidc':
        epsg = 3411 if hemisphere == 'north' else 3412
    elif grid_proj == 'ease':
        epsg = 6931 if hemisphere == 'north' else 6932
    else:
        raise ValueError('grid_proj must be "nsidc" or "ease"')

    crs = CRS.from_epsg(epsg)

    # ---- Recommended: use standard NSIDC extents ----
    # Example for EASE-Grid 2.0 SH; adjust if needed
    if epsg in (6931, 6932):
        half_width = 720 * resolution / 2.0
        area_extent = (-half_width, -half_width,
                         half_width,  half_width)
        xdim = ydim = 720
    else:
        # legacy NSIDC stereographic – use documented extents
        half_width = 3950e3
        area_extent = (-half_width, -half_width,
                         half_width,  half_width)
        xdim = int(round((area_extent[2]-area_extent[0]) / resolution))
        ydim = int(round((area_extent[3]-area_extent[1]) / resolution))
    
    proj_str = crs.to_proj4()

    area_def = AreaDefinition(f'{grid_proj}_{hemisphere}',crs.name,crs.srs,proj_str,xdim,ydim,area_extent)

    lon, lat = area_def.get_lonlats()
    x, y     = area_def.get_proj_coords()

    return lat, lon, x, y


if __name__ == '__main__':
    import numpy as np
    import timeit
    import os
    from pyresample.geometry import AreaDefinition
    import datetime
    import timeit
    import xarray as xr
    from pyproj import CRS
    import pandas as pd

    timeit_t0 = timeit.default_timer()

    #parameters and directories
    hemisphere       = 'south'
    grid_proj        = 'nsidc'
    amsre_conversion = True
    grid_res         = 12.5
    sample_radius    = 12.5

    dir_mask         = '/home/waynedj/Data/masks/nsidc/amsr2/'
    dir_input        = '/home/waynedj/Data/intermediate/ecice/'
    dir_output       = '/home/waynedj/Data/intermediate/ecice/netcdf/'

    list_of_files = os.listdir(dir_input)
    list_of_files = ['GW1AM2_201907261709_107A_L1SGBTBR_2220220-intermediate-v001-ECICE.txt']


    # fetch grid latlon and xy coordinates
    lats, lons, xc, yc = get_grids(hemisphere, grid_proj, grid_res)
    grid_shape         = lats.shape
    # give xc and yc variables a time dimension for dataset dim compatibility
    xc  = xc[np.newaxis,:,:]
    yc  = yc[np.newaxis,:,:]

    #               0     1      2     3      4    5      6       7      8      9     10    11   12   13     14       15     16    17   18   19    20     21       22     23      24      25       26      27      28       29      30
    #headers = [ORLINE,ORCELL,LATITD,LONGTD,BKAS,BRT19H,BRT19V,BRT22H,BRT22V,BRT37H,BRT37V,PR19,PR22,PR37,GD37v19v,GR22v19v,LAND,C_OW,C_FYI,C_YI,C_MYI,C_TOTAL,CONF_OW,CONF_I1,CONF_I2,CONF_I3,CONF_VEC,ITERAVG,OW_FILT,GR_FILT,CLOUD_FILT]
    for file in list_of_files:
        
        if '-ECICE' in file:
            print('Converting ' + file + ' from txt to nc')
            # fetch data from intermediate .txt file
            data      = np.genfromtxt(dir_input+file, skip_header=1, usecols=[2,3,5,17,18,19,20,21,22,23,24,25,26])
    #             0       1      2     3    4    5      6     7      8        9      10      11        12   
    #headers = [LATITD,LONGTD,BRT19H,C_OW,C_FYI,C_YI,C_MYI,C_TOTAL,CONF_OW,CONF_I1,CONF_I2,CONF_I3,CONF_VEC]
    
            latitude  = data[:,0].astype(int).reshape(grid_shape[0], grid_shape[1])
            longitude = data[:,1].astype(int).reshape(grid_shape[0], grid_shape[1])
            check_nan = data[:,2].reshape(grid_shape[0], grid_shape[1])
            OW        = data[:,3].reshape(grid_shape[0], grid_shape[1])
            FYI       = data[:,4].reshape(grid_shape[0], grid_shape[1])
            YI        = data[:,5].reshape(grid_shape[0], grid_shape[1])
            MYI       = data[:,6].reshape(grid_shape[0], grid_shape[1])
            TI        = data[:,7].reshape(grid_shape[0], grid_shape[1])
            CONF_OW   = data[:,8].reshape(grid_shape[0], grid_shape[1])
            CONF_I1   = data[:,9].reshape(grid_shape[0], grid_shape[1])
            CONF_I2   = data[:,10].reshape(grid_shape[0], grid_shape[1])
            CONF_I3   = data[:,11].reshape(grid_shape[0], grid_shape[1])
            CONF_VEC  = data[:,12].reshape(grid_shape[0], grid_shape[1])
            

            # uses and arbitary frequency channel to check if nan is returned (therefore no data retrieval)
            ind0          = np.isnan(check_nan)
            OW[ind0]      = np.nan
            FYI[ind0]     = np.nan
            YI[ind0]      = np.nan
            MYI[ind0]     = np.nan
            TI[ind0]      = np.nan
            CONF_I1[ind0] = np.nan
            CONF_I2[ind0] = np.nan
            CONF_I3[ind0] = np.nan
            CONF_OW[ind0] = np.nan
            CONF_VEC[ind0] = np.nan

            # ecice built-in open-water filter attributes 102 SIC in regions of open water
            ind1          = np.where(TI==102)
            OW[ind1]      = 100
            FYI[ind1]     = 0
            YI[ind1]      = 0
            MYI[ind1]     = 0
            TI[ind1]      = 0
            CONF_I1[ind1] = np.nan
            CONF_I2[ind1] = np.nan
            CONF_I3[ind1] = np.nan
            CONF_OW[ind1] = np.nan
            CONF_VEC[ind1] = np.nan

            # add a third dimension to dataarray for time
            OW      = OW[np.newaxis,:,:]
            YI      = YI[np.newaxis,:,:]
            FYI     = FYI[np.newaxis,:,:]
            MYI     = MYI[np.newaxis,:,:]
            TI      = TI[np.newaxis,:,:]
            CONF_I1 = CONF_I1[np.newaxis,:,:]
            CONF_I2 = CONF_I2[np.newaxis,:,:]
            CONF_I3 = CONF_I3[np.newaxis,:,:]
            CONF_OW = CONF_OW[np.newaxis,:,:]
            CONF_VEC= CONF_VEC[np.newaxis,:,:]
            
            # generate time array
            year   = int(file[7:11])
            month  = int(file[11:13])
            day    = int(file[13:15])
            hour   = int(file[15:17])
            minute = int(file[17:19])
            second = 0
            t      = np.array([datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)])
            
            # write to NETCDF4 file
            ds = xr.Dataset(
                data_vars = dict(
                    xc      = (['time','x','y',], xc),
                    yc      = (['time','x','y',], yc),
                    OW      = (['time','x','y',], OW),
                    YI      = (['time','x','y',], YI),
                    FYI     = (['time','x','y',],FYI),
                    MYI     = (['time','x','y',],MYI),
                    TI      = (['time','x','y',], TI),
                    CONF_I1 = (['time','x','y',], CONF_I1),
                    CONF_I2 = (['time','x','y',], CONF_I2),
                    CONF_I3 = (['time','x','y',], CONF_I3),
                    CONF_OW = (['time','x','y',], CONF_OW),
                    CONF_VEC= (['time','x','y',], CONF_VEC)),
                coords = dict(
                    longitude= (['x','y'],lons),
                    latitude = (['x','y'],lats),
                    time     = t),
                attrs = dict(
                    Title               = 'ECICE applied to AMSR-2 swaths',
                    Description         = 'Level 1B brightness temperatures from the AMSR-2 radiometer are processed with the ECICE sea ice type algorithm using BRT19H, GD37v19v, BRT37H, BRT37V frequency channels and regridded over entire polar region.',
                    Hemisphere          = hemisphere,
                    Grid_resolution     = str(grid_res) + ' km',
                    Grid_projection     = grid_proj,
                    Author              = 'Wayne de Jager, Department of Oceanography, University of Cape Town, South Africa',
                    DistVersion         = 'v0.0.1')
                    )
            
            ds.xc.attrs['long_name']       = 'x coordinate of projection (eastings)'
            ds.xc.attrs['units']           = 'm'
            ds.yc.attrs['long_name']       = 'y coordinate of projection (northings)'
            ds.yc.attrs['units']           = 'm'
            ds.OW.attrs['long_name']       = 'Open Water concentration'
            ds.OW.attrs['units']           = '%'
            ds.YI.attrs['long_name']       = 'Young Ice concentration'
            ds.YI.attrs['units']           = '%'
            ds.FYI.attrs['long_name']      = 'First Year Ice concentration'
            ds.FYI.attrs['units']          = '%'
            ds.MYI.attrs['long_name']      = 'Multi Year Ice concentration'
            ds.MYI.attrs['units']          = '%'
            ds.TI.attrs['long_name']       = 'Total Ice concentration'
            ds.TI.attrs['units']           = '%'
            ds.CONF_I1.attrs['long_name']  = 'Confidence level for First Year Ice'
            ds.CONF_I1.attrs['units']      = 'arbitrary units'
            ds.CONF_I2.attrs['long_name']  = 'Confidence level for Young Ice'
            ds.CONF_I2.attrs['units']      = 'arbitrary units'
            ds.CONF_I3.attrs['long_name']  = 'Confidence level for Multi Year Ice'
            ds.CONF_I3.attrs['units']      = 'arbitrary units'
            ds.CONF_OW.attrs['long_name']  = 'Confidence level for Open Water'
            ds.CONF_OW.attrs['units']      = 'arbitrary units'
            ds.CONF_VEC.attrs['long_name'] = 'Confidence level vector magnitude'
            ds.CONF_VEC.attrs['units']     = 'arbitrary units'



            ds.to_netcdf(dir_output+file[0:41]+file[59:65]+'.nc', format='NETCDF4')
            

        else:
            continue





    timeit_t1 = timeit.default_timer()

    print('[dt] Processing time: ' + str(np.round((timeit_t1-timeit_t0),3))+ 's')

# %%


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


extent = [310, 340, -80, -60]
proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)
fig, axes = plt.subplots(1, 2, figsize=(14, 7),subplot_kw={'projection': proj},constrained_layout=True)


yi = YI[0, :, :]
conf_i2 = CONF_VEC[0, :, :]

# --- Left: Young Ice concentration (percent) ---
vmin_yi = 0
vmax_yi = 100
im1 = axes[0].pcolormesh(lons, lats, yi, cmap='viridis', vmin=vmin_yi, vmax=vmax_yi,
                         shading='auto', transform=ccrs.PlateCarree())
axes[0].coastlines(resolution='50m')
axes[0].add_feature(cfeature.LAND, facecolor='lightgray')
axes[0].set_extent(extent, crs=ccrs.PlateCarree())
axes[0].set_title('Young Ice (YI)')
cb1 = plt.colorbar(im1, ax=axes[0], orientation='vertical', label='Concentration [%]')

# --- Right: Confidence for Young Ice (CONF_I2) ---
vmin_c = 0
vmax_c = 1
im2 = axes[1].pcolormesh(lons, lats, conf_i2, cmap='magma', vmin=vmin_c, vmax=vmax_c,
                         shading='auto', transform=ccrs.PlateCarree())
axes[1].coastlines(resolution='50m')
axes[1].add_feature(cfeature.LAND, facecolor='lightgray')
axes[1].set_extent(extent, crs=ccrs.PlateCarree())
axes[1].set_title('Confidence (CONF_I2)')
cb2 = plt.colorbar(im2, ax=axes[1], orientation='vertical', label='Confidence (arb. units)')

plt.show()


# %%
