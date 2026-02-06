
#%%

if __name__ == '__main__':
    import numpy as np
    import h5py
    import timeit
    import pyresample as pr
    import os
    import random

    timeit_t0 = timeit.default_timer()

    dir_amsr2        = '/home/waynedj/Data/amsr2/l1r/CaseStudy2019/'
    dir_output       = '/home/waynedj/Data/intermediate/ecice/CaseStudy2019/'
    n_sample         = 50
    resample_res     = 'res10'

    list_of_files = os.listdir(dir_amsr2)
    #list_of_files = ['GW1AM2_202207200118_194D_L1SGRTBR_2220220.h5']

    #select a subset of n_sample files from all AMSR2 swaths between 15-31 July, 2019, to reduce resource requirements 
    if len(list_of_files) > n_sample:
        list_of_files = random.sample(list_of_files, n_sample)

    for file in list_of_files:
        print('Preparing swath filename: ' + file)
        with h5py.File(dir_amsr2 + file, mode='r') as H:
            # fetch brightness temperatures from low frequency channels
            tb_19_h  = H[f'Brightness Temperature ({resample_res},18.7GHz,H)'][:]/100
            tb_19_v  = H[f'Brightness Temperature ({resample_res},18.7GHz,V)'][:]/100
            tb_22_h  = H[f'Brightness Temperature ({resample_res},23.8GHz,H)'][:]/100
            tb_22_v  = H[f'Brightness Temperature ({resample_res},23.8GHz,V)'][:]/100
            tb_37_h  = H[f'Brightness Temperature ({resample_res},36.5GHz,H)'][:]/100
            tb_37_v  = H[f'Brightness Temperature ({resample_res},36.5GHz,V)'][:]/100
            # generate latlon grid compatible with low frequency channels by modifying available 89GHz grid
            lat_a = H['Latitude of Observation Point for 89A'][:]
            lat   = lat_a[:,1::2]
            lon_a = H['Longitude of Observation Point for 89A'][:]
            lon   = lon_a[:,1::2]
            # generate indexing for ECICE
            grid_shape   = tb_19_h.shape
            grid_size    = grid_shape[0]*grid_shape[1]
            lines, cells = np.meshgrid(range(grid_shape[1]), range(grid_shape[0]))
            # generate empty backscattering array (backscattering not used for swath ECICE but algorithm still expects input)
            bks          = np.zeros(grid_size)
            # compute auxilary variables
            pr19         = (tb_19_v - tb_19_h)/(tb_19_v + tb_19_h)
            pr22         = (tb_22_v - tb_22_h)/(tb_22_v + tb_22_h)
            pr37         = (tb_37_v - tb_37_h)/(tb_37_v + tb_37_h)
            gr37v19v     = (tb_37_v - tb_19_v)/(tb_37_v + tb_19_v)
            gr22v19v     = (tb_22_v - tb_19_v)/(tb_22_v + tb_19_v)
            # generate land mask
            land         = H['Land_Ocean Flag 6 to 36'][:][0]
            #close h5 file
            H.close()
            # Flatten lat once (curvilinear is fine; we just need elementwise comparison)
            lat1         = lat.reshape(-1)
            # Mask: True where we want to blank out (lat > -55)
            mask         = lat1 > -55
            # Helper to blank out values where mask is True
            def nan_where(x):
                x1 = x.reshape(-1).astype(float)  # ensure float so NaN is allowed
                x1[mask] = np.nan
                return x1

            out = np.c_[
                lines.reshape(-1),          #0
                cells.reshape(-1),          #1
                lat1,                       #2
                lon.reshape(-1),            #3
                nan_where(bks),             #4
                nan_where(tb_19_h),         #5
                nan_where(tb_19_v),         #6
                nan_where(tb_22_h),         #7
                nan_where(tb_22_v),         #8
                nan_where(tb_37_h),         #9
                nan_where(tb_37_v),         #10
                nan_where(pr19),            #11
                nan_where(pr22),            #12
                nan_where(pr37),            #13
                nan_where(gr37v19v),        #14
                nan_where(gr22v19v),        #15
                nan_where(land),            #16
            ]
            
            headers    = ['ORLINE','ORCELL','LATITD','LONGTD','BKAS','BRT19H','BRT19V','BRT22H','BRT22V','BRT37H','BRT37V','PR19','PR22','PR37','GD37v19v','GR22v19v','LAND']
            formatting = ['%i'    ,'%i'    ,'%.2f'  ,'%.2f'  ,'%.2f','%.2f'  ,'%.2f'  ,'%.2f'  ,'%.2f'  ,'%.2f'  ,'%.2f'  ,'%.6f','%.6f','%.6f','%.6f'    ,'%.6f'    ,'%.2f']
            fname_out  = dir_output + file[0:-3] + '-intermediate-v001.txt'

            np.savetxt(
                fname_out,
                out,
                delimiter='\t',
                fmt=formatting,
                header='\t'.join(headers),
                comments='')

    timeit_t1 = timeit.default_timer()

    print('[dt] Processing time: ' + str(np.round((timeit_t1-timeit_t0),3))+ 's')

# %%
