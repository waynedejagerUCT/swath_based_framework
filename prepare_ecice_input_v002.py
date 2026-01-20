
#%%

if __name__ == '__main__':
    import numpy as np
    import h5py
    import timeit
    import pyresample as pr
    import os

    timeit_t0 = timeit.default_timer()

    dir_amsr2        = '/home/waynedj/Data/amsr2/l1r/'
    dir_output       = '/home/waynedj/Data/intermediate/ecice/'

    list_of_files = os.listdir(dir_amsr2)
    list_of_files = ['GW1AM2_201907261709_107A_L1SGRTBR_2220220.h5','GW1AM2_201907270213_203D_L1SGRTBR_2220220.h5']

    for file in list_of_files:
        print('Filename: ' + file)

        H        = h5py.File(dir_amsr2+file, mode='r')
        # fetch brightness temperatures from low frequency channels
        resample_res = 'res23'
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
            lat1,                       #2  (your comment says #3, but this is the 3rd column)
            lon.reshape(-1),            #3
            bks.reshape(-1),            #4
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
            land.reshape(-1)            #16
        ]
        
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
