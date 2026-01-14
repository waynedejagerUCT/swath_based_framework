# %%
'''
Author:
    Wayne de Jager
    Department of Oceanography, University of Cape Town

Purpose:
    prepare ECICE input files.

Details:
    The ECICE ice type algorithm requires text file input data. This python script does the following:
        - fetch the swath L1B AMSR-2 brightness temperatures in .hdf format from local machine.
        - generate empty NSIDC or EASE grid at custom resolution.
        - apply land mask to generated grid. Land mask must match resolution and hemisphere of generated grid.
            - proj projection parameters: https://www.spatialreference.org/ref/epsg/3411/ and https://www.spatialreference.org/ref/epsg/3412/
            - boundaries and grid sizes: https://nsidc.org/data/polar-stereo/ps_grids.html
            - investigate the plausibility of using EASE grid: https://pyresample.readthedocs.io/en/latest/geo_def.html#areadefinition
        - resample the swath AMSR-2 to the newly generated NSIDC grid.
        - reach variable gets reshaped from 2D array to 1D array, these 1D lists then form columns in table-like numpy array.
            - therefore each row is pixel data
        - an AMSR-E correction can be applied/ignored depending on boolean parameter.
            - This may be neccessary if distribution data was retreived from AMSR-E ([!] this needs to be checked)
        - other variables are created and appended to table as columns (eg gradient variables)
        - rows where mask==1 (ie land pixel) are removed from table to avoid uneccesary processing time.

Machine:
    - waynedj-legion5
    - Ubuntu 20.04.4 run via WSL2 on windows 11
    - conda env00...
'''

# %%
def get_mask(hemisphere):
    print('[f] get_mask(...)')
    if hemisphere[0:1] == 'n':
        fmask = dir_mask + 'amsr_gsfc_12n.hdf'
        # 1 for land, 0 for water
    elif hemisphere[0:1] == 's':
        fmask = dir_mask + 'amsr_nic_12s.hdf'
        # 1 for land, 0 for water, 2 for coastal zone
    else:
        print('ERROR: Unknown hemisphere parameter. Use "n" or "s" for northern or southern hemisphere respectively.')
    
    dmask = gdal.Open(fmask, gdal.GA_ReadOnly)
    mask  = dmask.ReadAsArray()

# %%
def get_grids(hemisphere, grid_proj, grid_res):
    '''
    Purpose:
        Generates polar stereographic grid in NSIDC or EASE projection
    Details:
        NSIDC:
            - NSIDC Sea Ice Polar Stereographic North (EPSG:3411): https://epsg.io/3411
            - NSIDC Sea Ice Polar Stereographic South (EPSG:3412): https://epsg.io/3412
            - WGS 84 / NSIDC EASE-Grid 2.0 North (EPSG:6931): https://epsg.io/6931
            - WGS 84 / NSIDC EASE-Grid 2.0 South (EPSG:6932): https://epsg.io/6932


        if coordinates string given as 'latlon', returns latitude, longitude in degrees
        if coordinates string given as 'xy', returns x-coord, y-coord in meters
    '''
    print('[f] get_grids(...)')

    resolution = grid_res*1000
    proj_id    = grid_proj+'_'+hemisphere[0:1]+'h'

    if proj_id == 'nsidc_nh':
        area_id      = proj_id
        discription  = 'NSIDC Sea Ice Polar Stereographic North (EPSG:3411)'
        epsg_code    = 'EPSG:3411'
        proj4_string = '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs'
        ar           = [i*1000 for i in [-3850,-5350,3750,5850]]
    elif proj_id == 'nsidc_sh':
        area_id      = proj_id
        discription  = 'NSIDC Sea Ice Polar Stereographic South (EPSG:3412)'
        epsg_code    = 'EPSG:3412'
        proj4_string = '+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs'
        ar           = [i*1000 for i in [-3950,-3950,3950,4350]]
    elif proj_id == 'ease_nh':
        area_id      = proj_id
        discription  = 'WGS 84 / NSIDC EASE-Grid 2.0 North (EPSG:6931)'
        epsg_code    = 'EPSG:6931'
        proj4_string = '+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
        ar           = [i*1000 for i in [-3850,-5350,3750,5850]]
    elif proj_id == 'ease_sh':
        area_id      = proj_id
        discription  = 'WGS 84 / NSIDC EASE-Grid 2.0 South (EPSG:6932)'
        epsg_code    = 'EPSG:6932'
        proj4_string = '+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
        ar           = [i*1000 for i in [-3950,-3950,3950,4350]]
    else:
        print('ERROR: could not identify grid projection. Please use "nsidc" or "ease" arguments.')


    grid_size  = lambda res:[int(a/res) for a in [(ar[2]-ar[0]),(ar[3]-ar[1])]]
    xdim, ydim = grid_size(resolution)

    area_def   = AreaDefinition(area_id, discription, proj_id, proj4_string, xdim, ydim, ar)
    lon, lat   = area_def.get_lonlats()
    x_grid     = (np.arange(ar[0]+resolution/2.,ar[2],resolution))
    y_grid     = (np.arange(ar[1]+resolution/2.,ar[3],resolution))
    xc, yc     = np.meshgrid(x_grid,y_grid)

    return lat, lon, xc, yc

# %%
def swath_remapping(in_lat, in_lon, in_data, hemisphere, grid_proj, grid_res, sample_radius):
    print('[f] swath_remapping(...)')

    sample_radius = sample_radius*1000 #units km to meters

    # input grid: input swath data
    in_grid  = pr.geometry.SwathDefinition(lons=pr.utils.wrap_longitudes(in_lon), lats=in_lat)
    # output grid: generate custom grid on which swath data will be remapped
    out_lat, out_lon, out_xc, out_yc = get_grids(hemisphere, grid_proj, grid_res)
    out_grid = pr.geometry.SwathDefinition(lons=out_lon, lats=out_lat)

    # remap input_grid onto output_grid
    grdata   = pr.kd_tree.resample_nearest(in_grid, in_data, out_grid, radius_of_influence=sample_radius, fill_value=np.nan)

    return out_lat, out_lon, out_xc, out_yc, grdata

# %%
if __name__ == '__main__':
    import numpy as np
    import h5py
    import timeit
    import pyresample as pr
    import datetime
    import os
    from pyresample import get_area_def
    from pyresample.geometry import AreaDefinition
    #from osgeo import gdal

    timeit_t0 = timeit.default_timer()

    #parameters and directories
    hemisphere       = 'south'
    grid_proj        = 'nsidc'
    amsre_conversion = True
    grid_res         = 12.5
    sample_radius    = 12.5

    dir_mask         = '/home/waynedj/Data/masks/nsidc/amsr2/'
    dir_amsr2        = '/home/waynedj/Data/amsr2/l1b/'
    dir_output       = '/home/waynedj/Data/intermediate/ecice/'

    list_of_files = os.listdir(dir_amsr2)
    list_of_files = ['GW1AM2_201907261709_107A_L1SGBTBR_2220220.h5']

    for file in list_of_files:
        print('Filename: ' + file)

        H        = h5py.File(dir_amsr2+file, mode='r')
        # fetch brightness temperatures from low frequency channels
        tb_19_h  = H['Brightness Temperature (18.7GHz,H)'][:]/100
        tb_19_v  = H['Brightness Temperature (18.7GHz,V)'][:]/100
        tb_22_h  = H['Brightness Temperature (23.8GHz,H)'][:]/100
        tb_22_v  = H['Brightness Temperature (23.8GHz,V)'][:]/100
        tb_37_h  = H['Brightness Temperature (36.5GHz,H)'][:]/100
        tb_37_v  = H['Brightness Temperature (36.5GHz,V)'][:]/100
        # generate latlon grid compatible with low frequency channels by modifying available 89GHz grid
        lat_a    = H['Latitude of Observation Point for 89A'][:]
        in_lat   = lat_a[:,1::2]
        lon_a    = H['Longitude of Observation Point for 89B'][:]
        in_lon   = lon_a[:,1::2]
        # remap swath brightness temperatures onto custom grid
        out_lat, out_lon, out_xc, out_yc, TB_19_H = swath_remapping(in_lat, in_lon, tb_19_h, hemisphere, grid_proj, grid_res, sample_radius)
        out_lat, out_lon, out_xc, out_yc, TB_19_V = swath_remapping(in_lat, in_lon, tb_19_v, hemisphere, grid_proj, grid_res, sample_radius)
        out_lat, out_lon, out_xc, out_yc, TB_22_H = swath_remapping(in_lat, in_lon, tb_22_h, hemisphere, grid_proj, grid_res, sample_radius)
        out_lat, out_lon, out_xc, out_yc, TB_22_V = swath_remapping(in_lat, in_lon, tb_22_v, hemisphere, grid_proj, grid_res, sample_radius)
        out_lat, out_lon, out_xc, out_yc, TB_37_H = swath_remapping(in_lat, in_lon, tb_37_h, hemisphere, grid_proj, grid_res, sample_radius)
        out_lat, out_lon, out_xc, out_yc, TB_37_V = swath_remapping(in_lat, in_lon, tb_37_v, hemisphere, grid_proj, grid_res, sample_radius)
        # generate indexing for ECICE
        grid_shape   = TB_19_H.shape
        grid_size    = grid_shape[0]*grid_shape[1]
        lines, cells = np.meshgrid(range(grid_shape[1]), range(grid_shape[0]))
        # generate empty backscattering array (backscattering not used for swath ECICE but algorithm still expects input)
        bks          = np.zeros(grid_size)
        # compute auxilary variables
        pr19         = (TB_19_V - TB_19_H)/(TB_19_V + TB_19_H)
        pr22         = (TB_22_V - TB_22_H)/(TB_22_V + TB_22_H)
        pr37         = (TB_37_V - TB_37_H)/(TB_37_V + TB_37_H)
        gr37v19v     = (TB_37_V - TB_19_V)/(TB_37_V + TB_19_V)
        gr22v19v     = (TB_22_V - TB_19_V)/(TB_22_V + TB_19_V)
        # generate land mask
        land         = np.zeros(grid_size)
        # compile main output table for .txt file
        out = np.c_[
            lines.reshape(-1),     #0
            cells.reshape(-1),     #1
            out_lat.reshape(-1),   #3   <<< check variable, currently not compatible with ease grid
            out_lon.reshape(-1),   #4   <<< check variable, currently not compatible with ease grid
            bks.reshape(-1),       #5
            TB_19_H.reshape(-1),   #6
            TB_19_V.reshape(-1),   #7
            TB_22_H.reshape(-1),   #8
            TB_22_V.reshape(-1),   #9            
            TB_37_H.reshape(-1),   #10
            TB_37_V.reshape(-1),   #11           
            pr19.reshape(-1),      #12
            pr22.reshape(-1),      #13
            pr37.reshape(-1),      #14
            gr37v19v.reshape(-1),  #15
            #gr37v19v.reshape(-1),  #16 note the duplicate. Error in ecice indexing means this might be neccessary
            gr22v19v.reshape(-1),  #17
            land.reshape(-1)]      #18
        
        # save data to .txt file
        fname_out = dir_output + file[0:-3] + '-intermediate-v001.txt'
        np.savetxt(fname_out, out, delimiter='\t', fmt=['%i','%i','%.2f','%.2f','%.2f','%.2f','%.2f','%.2f','%.2f','%.2f','%.2f','%.6f','%.6f','%.6f','%.6f','%.6f','%.2f'])
        # generate an empty temporary .txt file
        fname_tmp = dir_output + file[0:-3] + '-intermediate-v001-tmp.txt'
        # open data and temporary .txt files
        file_orig = open(fname_out,'r')
        file_tmp  = open(fname_tmp,'w')
        # define column headers
        headers = ['ORLINE','ORCELL','LATITD','LONGTD','BKAS','BRT19H','BRT19V','BRT22H','BRT22V','BRT37H','BRT37V','PR19','PR22','PR37','GD37v19v','GR22v19v','LAND']
        
        # write column headers as first row in temporary .txt file
        file_tmp.write('\t'.join(headers)+'\n')
        # iterate through each row in data .txt file and append row to temporary .txt file
        for line in file_orig.readlines():
            file_tmp.write(line)
        # close both data and tempory .txt files
        file_orig.close()
        file_tmp.close()
        # remove data .txt file and rename temporary .txt file to original file name
        os.remove(fname_out)
        os.rename(fname_tmp, fname_out)



    timeit_t1 = timeit.default_timer()

    print('[dt] Processing time: ' + str(np.round((timeit_t1-timeit_t0),3))+ 's')


# %%
