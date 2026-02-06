
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

def get_grids(grid_res, grid_proj):
    import xarray as xr
    import numpy as np
    from pyresample.geometry import GridDefinition

    if grid_proj == 'EASE2':
        fname = f"/home/waynedj/Data/grids/NSIDC0772_LatLon_EASE2_S{grid_res}km_v1.1.nc"
    elif grid_proj == 'PS':
        fname = f"/home/waynedj/Data/grids/NSIDC0771_LatLon_PS_S{grid_res}km_v1.1.nc"
    else:
        raise ValueError("Unsupported grid projection. Must be 'EASE2' or 'PS'.")

    ds = xr.open_dataset(fname)

    lat = np.nan_to_num(ds.latitude.data)
    lon = np.nan_to_num(ds.longitude.data)
    x   = ds.x.data
    y   = ds.y.data

    out_grid = GridDefinition(lons=lon, lats=lat)

    return out_grid, lat, lon, x, y

def swath_remapping(in_lat, in_lon, in_data, out_grid, sample_radius):

    roi_m = sample_radius * 1000 #units km to meters
    # input grid: input swath data
    in_grid  = pr.geometry.SwathDefinition(lons=pr.utils.wrap_longitudes(in_lon), lats=in_lat)
    # remap input_grid onto output_grid
    grdata   = pr.kd_tree.resample_nearest(in_grid, in_data, out_grid, radius_of_influence=roi_m, fill_value=np.nan)
    
    return grdata


if __name__ == '__main__':

    timeit_t0 = timeit.default_timer()

    #parameters and directories
    grid_res         = 12.5
    grid_proj       = 'PS'
    sample_radius    = grid_res*1.5

    dir_input        = '/home/waynedj/Data/intermediate/ecice/scale2022/'
    dir_output       = '/home/waynedj/Data/intermediate/ecice/scale2022/netcdf/'

    list_of_files = os.listdir(dir_input)
    #list_of_files = ['GW1AM2_201907261709_107A_L1SGRTBR_2220220-intermediate-v001-ECICE.txt', 'GW1AM2_201907270213_203D_L1SGRTBR_2220220-intermediate-v001-ECICE.txt']

    # fetch grid latlon and xy coordinates
    out_grid, lats, lons, xc, yc = get_grids(grid_res, grid_proj)
    xc, yc                       = np.meshgrid(xc, yc)
    grid_shape                   = lats.shape
    # give xc and yc variables a time dimension for dataset dim compatibility
    xc  = xc[np.newaxis,:,:]
    yc  = yc[np.newaxis,:,:]

    COL = dict(ORLINE=0,ORCELL=1,LAT=2,LON=3,BKAS=4,BRT19H=5,BRT19V=6,BRT22H=7,BRT22V=8,
               BRT37H=9,BRT37V=10,PR19=11,PR22=12,PR37=13,GD37v19v=14,GR22v19v=15,LAND=16,
               C_OW=17,C_FYI=18,C_YI=19,C_MYI=20,C_TOT=21,CONF_OW=22,CONF_I1=23,
               CONF_I2=24,CONF_I3=25,CONF_VEC=26)
    
    for file in list_of_files:
        if '-ECICE' not in file:
            continue

        print('Mapping ' + file + f' from txt to {grid_proj} at {grid_res}km grid in netcdf format.')
        # fetch data from intermediate .txt file
        data       = np.genfromtxt(os.path.join(dir_input, file), skip_header=1)
        # Unpack 1D columns
        ORLINE     = data[:, COL["ORLINE"]].astype(int)
        ORCELL     = data[:, COL["ORCELL"]].astype(int)
        LATITD_1   = data[:, COL["LAT"]].astype(float)
        LONGTD_1   = data[:, COL["LON"]].astype(float)
        LAND_1     = data[:, COL["LAND"]].astype(float)
        C_OW_1     = data[:, COL["C_OW"]].astype(float)
        C_FYI_1    = data[:, COL["C_FYI"]].astype(float)
        C_YI_1     = data[:, COL["C_YI"]].astype(float)
        C_MYI_1    = data[:, COL["C_MYI"]].astype(float)
        C_TOT_1    = data[:, COL["C_TOT"]].astype(float)
        CONF_OW_1  = data[:, COL["CONF_OW"]].astype(float)
        CONF_I1_1  = data[:, COL["CONF_I1"]].astype(float)
        CONF_I2_1  = data[:, COL["CONF_I2"]].astype(float)
        CONF_I3_1  = data[:, COL["CONF_I3"]].astype(float)
        CONF_VEC_1 = data[:, COL["CONF_VEC"]].astype(float)
        # Rebuild native swath 2D arrays using ORLINE/ORCELL (order independent)
        swath_lat  = scatter_to_grid(ORLINE, ORCELL, LATITD_1, fill=np.nan, dtype=float)
        swath_lon  = scatter_to_grid(ORLINE, ORCELL, LONGTD_1, fill=np.nan, dtype=float)
        LAND       = scatter_to_grid(ORLINE, ORCELL, LAND_1, fill=np.nan, dtype=float)
        OW         = scatter_to_grid(ORLINE, ORCELL, C_OW_1, fill=np.nan, dtype=float)
        FYI        = scatter_to_grid(ORLINE, ORCELL, C_FYI_1, fill=np.nan, dtype=float)
        YI         = scatter_to_grid(ORLINE, ORCELL, C_YI_1, fill=np.nan, dtype=float)
        MYI        = scatter_to_grid(ORLINE, ORCELL, C_MYI_1, fill=np.nan, dtype=float)
        TI         = scatter_to_grid(ORLINE, ORCELL, C_TOT_1, fill=np.nan, dtype=float)
        CONF_OW    = scatter_to_grid(ORLINE, ORCELL, CONF_OW_1, fill=np.nan, dtype=float)
        CONF_I1    = scatter_to_grid(ORLINE, ORCELL, CONF_I1_1, fill=np.nan, dtype=float)
        CONF_I2    = scatter_to_grid(ORLINE, ORCELL, CONF_I2_1, fill=np.nan, dtype=float)
        CONF_I3    = scatter_to_grid(ORLINE, ORCELL, CONF_I3_1, fill=np.nan, dtype=float)
        CONF_VEC   = scatter_to_grid(ORLINE, ORCELL, CONF_VEC_1, fill=np.nan, dtype=float)

        # Handle ECICE open-water filter (TI==102)
        ind1           = np.where(TI == 102)
        OW[ind1]       = 100
        FYI[ind1]      = 0
        YI[ind1]       = 0
        MYI[ind1]      = 0
        TI[ind1]       = 0
        CONF_I1[ind1]  = np.nan
        CONF_I2[ind1]  = np.nan
        CONF_I3[ind1]  = np.nan
        CONF_OW[ind1]  = np.nan
        CONF_VEC[ind1] = np.nan
        
        # Resample swath -> EASE2 grid
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
        LAND_g     = swath_remapping(swath_lat, swath_lon, LAND, out_grid, sample_radius)
                    
        # Add time dimension
        OW_g, FYI_g, YI_g, MYI_g, TI_g, CONF_I1_g, CONF_I2_g, CONF_I3_g, CONF_OW_g, CONF_VEC_g, LAND_g = [a[np.newaxis, :, :] for a in (OW_g, FYI_g, YI_g, MYI_g, TI_g, CONF_I1_g, CONF_I2_g, CONF_I3_g, CONF_OW_g, CONF_VEC_g, LAND_g)]

        # Parse time from filename format: GW1AM2_YYYYMMDDhhmm_...
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
                Title='ECICE applied to AMSR-2 swaths. Native L1R swath data at 23km spatial resolution.',
                Description=f'ECICE outputs reconstructed on native ORLINE/ORCELL swath grid, then resampled to {grid_proj}.',
                Hemisphere='Southern Hemisphere',
                Grid_resolution=str(grid_res) + ' km',
                Grid_projection=grid_proj,
                Author='Wayne de Jager, Department of Oceanography, University of Cape Town, South Africa',
                DistVersion='v0.0.2'
            )
        )

        # Save dataset to netcdf
        ds.to_netcdf(dir_output + file[0:41] + file[59:65] + '.nc', format='NETCDF4')



    timeit_t1 = timeit.default_timer()

    print('[dt] Processing time: ' + str(np.round((timeit_t1-timeit_t0),3))+ 's')





# %%
