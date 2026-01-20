
#%%

import numpy as np
import timeit
import os
import datetime
import timeit
import xarray as xr
import pyresample as pr
import numpy as np

def scatter_to_grid(orline_1d, orcell_1d, values_1d, fill=np.nan, dtype=float):
    """
    Reconstruct a 2D swath grid from 1D vectors using ORLINE/ORCELL indexing.
    Works regardless of ordering and with curvilinear grids.
    """
    orline = orline_1d.astype(int)
    orcell = orcell_1d.astype(int)

    n_lines = orline.max() + 1
    n_cells = orcell.max() + 1

    grid = np.full((n_lines, n_cells), fill, dtype=dtype)
    grid[orline, orcell] = values_1d.astype(dtype)
    return grid

def get_grids(grid_res):
    import xarray as xr
    import numpy as np
    from pyresample.geometry import GridDefinition

    fname = f"/home/waynedj/Data/grids/ease2/NSIDC0772_LatLon_EASE2_S{grid_res}km_v1.1.nc"
    ds = xr.open_dataset(fname)

    lat = np.nan_to_num(ds.latitude.data)
    lon = np.nan_to_num(ds.longitude.data)
    x   = ds.x.data
    y   = ds.y.data

    out_grid = GridDefinition(lons=lon, lats=lat)

    return out_grid, lat, lon, x, y

def swath_remapping(in_lat, in_lon, in_data, out_grid, sample_radius):

    sample_radius = sample_radius*1000 #units km to meters
    # input grid: input swath data
    in_grid  = pr.geometry.SwathDefinition(lons=pr.utils.wrap_longitudes(in_lon), lats=in_lat)
    # remap input_grid onto output_grid
    grdata   = pr.kd_tree.resample_nearest(in_grid, in_data, out_grid, radius_of_influence=sample_radius, fill_value=np.nan)

    return grdata


if __name__ == '__main__':

    timeit_t0 = timeit.default_timer()

    #parameters and directories
    amsre_conversion = True
    grid_res         = 12.5
    sample_radius    = grid_res*1.5

    dir_input        = '/home/waynedj/Data/intermediate/ecice/'
    dir_output       = '/home/waynedj/Data/intermediate/ecice/netcdf/'

    list_of_files = os.listdir(dir_input)
    list_of_files = ['GW1AM2_201907261709_107A_L1SGRTBR_2220220-intermediate-v001-ECICE.txt', 'GW1AM2_201907270213_203D_L1SGRTBR_2220220-intermediate-v001-ECICE.txt']

    # fetch grid latlon and xy coordinates
    out_grid, lats, lons, xc, yc = get_grids(grid_res)
    xc, yc                       = np.meshgrid(xc, yc)
    grid_shape                   = lats.shape
    # give xc and yc variables a time dimension for dataset dim compatibility
    xc  = xc[np.newaxis,:,:]
    yc  = yc[np.newaxis,:,:]

    #               0     1      2     3      4    5      6       7      8      9     10    11   12   13     14       15     16    17   18   19    20     21       22     23      24      25       26      27      28       29      30
    #headers = [ORLINE,ORCELL,LATITD,LONGTD,BKAS,BRT19H,BRT19V,BRT22H,BRT22V,BRT37H,BRT37V,PR19,PR22,PR37,GD37v19v,GR22v19v,LAND,C_OW,C_FYI,C_YI,C_MYI,C_TOTAL,CONF_OW,CONF_I1,CONF_I2,CONF_I3,CONF_VEC,ITERAVG,OW_FILT,GR_FILT,CLOUD_FILT]
    
    usecols = [
    0, 1,   # ORLINE, ORCELL (needed to rebuild 2D swath)
    2, 3,   # LATITD, LONGTD
    16,     # LAND
    17, 18, 19, 20, 21,  # C_OW, C_FYI, C_YI, C_MYI, C_TOTAL
    22, 23, 24, 25,      # CONF_OW, CONF_I1, CONF_I2, CONF_I3
    26                   # CONF_VEC
    ]
    
    for file in list_of_files:
        
        if '-ECICE' in file:
            print('Converting ' + file + ' from txt to nc')
            # fetch data from intermediate .txt file
            data = np.genfromtxt(dir_input + file, skip_header=1, usecols=usecols)
            # Unpack 1D columns
            ORLINE   = data[:, 0].astype(int)
            ORCELL   = data[:, 1].astype(int)
            LATITD_1 = data[:, 2].astype(float)
            LONGTD_1 = data[:, 3].astype(float)
            LAND_1   = data[:, 4].astype(float)
            C_OW_1   = data[:, 5].astype(float)
            C_FYI_1  = data[:, 6].astype(float)
            C_YI_1   = data[:, 7].astype(float)
            C_MYI_1  = data[:, 8].astype(float)
            C_TOT_1  = data[:, 9].astype(float)
            CONF_OW_1  = data[:, 10].astype(float)
            CONF_I1_1  = data[:, 11].astype(float)
            CONF_I2_1  = data[:, 12].astype(float)
            CONF_I3_1  = data[:, 13].astype(float)
            CONF_VEC_1 = data[:, 14].astype(float)

            # Rebuild native swath 2D arrays using ORLINE/ORCELL (order independent)
            swath_lat  = scatter_to_grid(ORLINE, ORCELL, LATITD_1, fill=np.nan, dtype=float)
            swath_lon  = scatter_to_grid(ORLINE, ORCELL, LONGTD_1, fill=np.nan, dtype=float)
            LAND   = scatter_to_grid(ORLINE, ORCELL, LAND_1, fill=np.nan, dtype=float)
            OW     = scatter_to_grid(ORLINE, ORCELL, C_OW_1, fill=np.nan, dtype=float)
            FYI    = scatter_to_grid(ORLINE, ORCELL, C_FYI_1, fill=np.nan, dtype=float)
            YI     = scatter_to_grid(ORLINE, ORCELL, C_YI_1, fill=np.nan, dtype=float)
            MYI    = scatter_to_grid(ORLINE, ORCELL, C_MYI_1, fill=np.nan, dtype=float)
            TI     = scatter_to_grid(ORLINE, ORCELL, C_TOT_1, fill=np.nan, dtype=float)
            CONF_OW  = scatter_to_grid(ORLINE, ORCELL, CONF_OW_1, fill=np.nan, dtype=float)
            CONF_I1  = scatter_to_grid(ORLINE, ORCELL, CONF_I1_1, fill=np.nan, dtype=float)
            CONF_I2  = scatter_to_grid(ORLINE, ORCELL, CONF_I2_1, fill=np.nan, dtype=float)
            CONF_I3  = scatter_to_grid(ORLINE, ORCELL, CONF_I3_1, fill=np.nan, dtype=float)
            CONF_VEC = scatter_to_grid(ORLINE, ORCELL, CONF_VEC_1, fill=np.nan, dtype=float)

            # Optional: If you used a “check_nan” channel before, you can define one here.
            # Example: treat missing lat/lon as missing retrieval
            ind0 = np.isnan(swath_lat) | np.isnan(swath_lon)

            for arr in [OW, FYI, YI, MYI, TI, CONF_I1, CONF_I2, CONF_I3, CONF_OW, CONF_VEC]:
                arr[ind0] = np.nan

            # Handle ECICE open-water filter (TI==102)
            ind1 = np.where(TI == 102)
            OW[ind1] = 100
            FYI[ind1] = 0
            YI[ind1] = 0
            MYI[ind1] = 0
            TI[ind1] = 0
            CONF_I1[ind1] = np.nan
            CONF_I2[ind1] = np.nan
            CONF_I3[ind1] = np.nan
            CONF_OW[ind1] = np.nan
            CONF_VEC[ind1] = np.nan

            # --------------------------------------------------------------------
            # Resample swath -> EASE2 grid using your existing swath_remapping()
            # --------------------------------------------------------------------
            OW_g       = swath_remapping(swath_lat, swath_lon, OW,       out_grid, sample_radius)
            FYI_g      = swath_remapping(swath_lat, swath_lon, FYI,      out_grid, sample_radius)
            YI_g       = swath_remapping(swath_lat, swath_lon, YI,       out_grid, sample_radius)
            MYI_g      = swath_remapping(swath_lat, swath_lon, MYI,      out_grid, sample_radius)
            TI_g       = swath_remapping(swath_lat, swath_lon, TI,       out_grid, sample_radius)
            CONF_I1_g  = swath_remapping(swath_lat, swath_lon, CONF_I1,  out_grid, sample_radius)
            CONF_I2_g  = swath_remapping(swath_lat, swath_lon, CONF_I2,  out_grid, sample_radius)
            CONF_I3_g  = swath_remapping(swath_lat, swath_lon, CONF_I3,  out_grid, sample_radius)
            CONF_OW_g  = swath_remapping(swath_lat, swath_lon, CONF_OW,  out_grid, sample_radius)
            CONF_VEC_g = swath_remapping(swath_lat, swath_lon, CONF_VEC, out_grid, sample_radius)

            # LAND is categorical; nearest-neighbour is OK, but you may also keep LAND from your mask.
            LAND_g     = swath_remapping(swath_lat, swath_lon, LAND,     out_grid, sample_radius)
                        
            # Add time dimension
            OW_g       = OW_g[np.newaxis, :, :]
            FYI_g      = FYI_g[np.newaxis, :, :]
            YI_g       = YI_g[np.newaxis, :, :]
            MYI_g      = MYI_g[np.newaxis, :, :]
            TI_g       = TI_g[np.newaxis, :, :]
            CONF_I1_g  = CONF_I1_g[np.newaxis, :, :]
            CONF_I2_g  = CONF_I2_g[np.newaxis, :, :]
            CONF_I3_g  = CONF_I3_g[np.newaxis, :, :]
            CONF_OW_g  = CONF_OW_g[np.newaxis, :, :]
            CONF_VEC_g = CONF_VEC_g[np.newaxis, :, :]
            LAND_g     = LAND_g[np.newaxis, :, :]

            # Parse time from filename (keep your approach)
            year   = int(file[7:11])
            month  = int(file[11:13])
            day    = int(file[13:15])
            hour   = int(file[15:17])
            minute = int(file[17:19])
            second = 0
            t = np.array([datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)])

            # Create Dataset
            ds = xr.Dataset(
                data_vars=dict(
                    xc=(['time','x','y'], xc),
                    yc=(['time','x','y'], yc),
                    OW=(['time','x','y'], OW_g),
                    YI=(['time','x','y'], YI_g),
                    FYI=(['time','x','y'], FYI_g),
                    MYI=(['time','x','y'], MYI_g),
                    TI=(['time','x','y'], TI_g),
                    CONF_I1=(['time','x','y'], CONF_I1_g),
                    CONF_I2=(['time','x','y'], CONF_I2_g),
                    CONF_I3=(['time','x','y'], CONF_I3_g),
                    CONF_OW=(['time','x','y'], CONF_OW_g),
                    CONF_VEC=(['time','x','y'], CONF_VEC_g),
                    LAND=(['time','x','y'], LAND_g),
                ),
                coords=dict(
                    longitude=(['x','y'], lons),
                    latitude=(['x','y'], lats),
                    time=t
                ),
                attrs=dict(
                    Title='ECICE applied to AMSR-2 swaths',
                    Description='ECICE outputs reconstructed on native ORLINE/ORCELL swath grid, then resampled to EASE2.',
                    Hemisphere='Southern Hemisphere',
                    Grid_resolution=str(grid_res) + ' km',
                    Grid_projection='EASE2',
                    Author='Wayne de Jager, Department of Oceanography, University of Cape Town, South Africa',
                    DistVersion='v0.0.2'
                )
            )

            # Save dataset to netcdf
            ds.to_netcdf(dir_output + file[0:41] + file[59:65] + '.nc', format='NETCDF4')

            

        else:
            continue


    timeit_t1 = timeit.default_timer()

    print('[dt] Processing time: ' + str(np.round((timeit_t1-timeit_t0),3))+ 's')





# %%
