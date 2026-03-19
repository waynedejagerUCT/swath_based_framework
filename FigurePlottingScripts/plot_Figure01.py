
#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pandas as pd
import os
import timeit
from matplotlib import colormaps

timeit_t0 = timeit.default_timer()



def CustomColorMap1(cmap):
    # generate custom cmap
    boundaries = np.arange(0, 100, 5)
    base_cmap = colormaps[cmap].resampled(len(boundaries))
    colors = base_cmap(np.arange(len(boundaries)))
    cmap1 = mpl.colors.ListedColormap(colors, name="k")
    cmap1.set_over(colors[-1])  # set over-color to last color

    return cmap1

extent        = [-130000, 140000, 3440000, 3630000]
proj_ease     = ccrs.LambertAzimuthalEqualArea(central_latitude=-90,central_longitude=0)
proj_ps       = ccrs.SouthPolarStereo(central_longitude=0)
proj_pl       = ccrs.PlateCarree()
vmin          = 0
vmax          = 100
cmap          = 'jet'
track_density = 30  # in minutes


list_of_files = os.listdir('/home/waynedj/Data/intermediate/ecice/scale2022/netcdf/')
list_of_files = sorted(list_of_files)
locations     = pd.read_csv('/home/waynedj/Data/SCALE/data/track/SCALE-WIN22-SDS_1min.csv', parse_dates=['TIME_SERVER'])
locations     = locations.iloc[::track_density, :].reset_index(drop=True)  # downsample to 1 data point every 3 hours
# load grids
ds_grid_06250 = xr.open_dataset('/home/waynedj/Data/grids/NSIDC0771_LatLon_PS_S6.25km_v1.1.nc')
ds_grid_12500 = xr.open_dataset('/home/waynedj/Data/grids/NSIDC0771_LatLon_PS_S12.5km_v1.1.nc')
ds_grid_25000 = xr.open_dataset('/home/waynedj/Data/grids/NSIDC0771_LatLon_PS_S25km_v1.1.nc')
# create figure and axes
fig, axes = plt.subplots(2, 3, figsize=(22, 13), subplot_kw={'projection': proj_ps})
plt.subplots_adjust(wspace=0.05, hspace=0.1, bottom=0.18)

day = 22
ds_osisaf    = xr.open_dataset(f'/home/waynedj/Data/sic/osi408a/ice_conc_sh_polstere-100_amsr2_202207{day}1200.nc')
ds_asi       = xr.open_dataset(f'/home/waynedj/Data/sic/asi/asi-AMSR2-s6250-202207{day}-v5.4.nc')
ds_ecice     = xr.open_dataset(f'/home/waynedj/Data/sic/ecice/ECICE-IcetypesUncorrected-202207{day}.nc')
im_osisaf    = axes[0, 0].pcolormesh(ds_osisaf.lon.data, ds_osisaf.lat.data, ds_osisaf.ice_conc.data[0],cmap=CustomColorMap1(cmap),vmin=vmin, vmax=vmax,shading='nearest',transform=proj_pl)
im_asi       = axes[0, 1].pcolormesh(ds_grid_06250.longitude.data, ds_grid_06250.latitude.data, np.flipud(ds_asi.z.data),cmap=CustomColorMap1(cmap),vmin=vmin, vmax=vmax,shading='nearest',transform=proj_pl)
im_ecice     = axes[0, 2].pcolormesh(ds_grid_12500.longitude.data, ds_grid_12500.latitude.data, ds_ecice.TOTAL_ICE.data,cmap=CustomColorMap1(cmap),vmin=vmin, vmax=vmax,shading='nearest',transform=proj_pl)

#axes[0, 0].set_title('OSISAF', fontsize=20, pad=20)
#axes[0, 1].set_title('ASI', fontsize=20, pad=20)
#axes[0, 2].set_title('ECICE', fontsize=20, pad=20)

day = 23
ds_osisaf    = xr.open_dataset(f'/home/waynedj/Data/sic/osi408a/ice_conc_sh_polstere-100_amsr2_202207{day}1200.nc')
ds_asi       = xr.open_dataset(f'/home/waynedj/Data/sic/asi/asi-AMSR2-s6250-202207{day}-v5.4.nc')
ds_ecice     = xr.open_dataset(f'/home/waynedj/Data/sic/ecice/ECICE-IcetypesUncorrected-202207{day}.nc')
im_osisaf    = axes[1, 0].pcolormesh(ds_osisaf.lon.data, ds_osisaf.lat.data, ds_osisaf.ice_conc.data[0],cmap=CustomColorMap1(cmap),vmin=vmin, vmax=vmax,shading='nearest',transform=proj_pl)
im_asi       = axes[1, 1].pcolormesh(ds_grid_06250.longitude.data, ds_grid_06250.latitude.data, np.flipud(ds_asi.z.data),cmap=CustomColorMap1(cmap),vmin=vmin, vmax=vmax,shading='nearest',transform=proj_pl)
im_ecice     = axes[1, 2].pcolormesh(ds_grid_12500.longitude.data, ds_grid_12500.latitude.data, ds_ecice.TOTAL_ICE.data,cmap=CustomColorMap1(cmap),vmin=vmin, vmax=vmax,shading='nearest',transform=proj_pl)

# Add a shared colorbar at the bottom
left, bottom, width, height = 0.2, 0.10, 0.6, 0.04
new_width = 0.5
new_left = left - (new_width - width) / 2
cbar_ax = fig.add_axes([new_left, bottom, new_width, height])
cbar = fig.colorbar(im_ecice, cax=cbar_ax, orientation='horizontal', label=f'Daily Mean Sea Ice Concentration (%)')
cbar.ax.xaxis.label.set_size(20)
cbar.ax.tick_params(labelsize=14)


# add map details
label_text = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
date_text = ['OSISAF 22 July', 'ASI 22 July', r'ECICE$_{TOTAL}$ 22 July', 'OSISAF 23 July', 'ASI 23 July', r'ECICE$_{total}$ 23 July']
for i, ax in enumerate(axes.flat):
        ax.text(0.02, 0.97, label_text[i],transform=ax.transAxes,ha="left",va="top",fontsize=22,color="black",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="black", linewidth=0.7))
        ax.text(0.02, 0.02, date_text[i],transform=ax.transAxes,ha="left",va="bottom",fontsize=20,color="white",
                bbox=dict(boxstyle="square,pad=0.12", facecolor="black", edgecolor="black", linewidth=0.6))
        ax.set_extent(extent, crs=proj_ps)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='grey', alpha=1, linestyle='-')
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = True
        gl.bottom_labels = True
        gl.xlabel_style = {'color': 'k', 'fontsize': 12}
        gl.ylabel_style = {'color': 'k', 'fontsize': 12}


ref_time_0 = pd.Timestamp(year=2022, month=7, day=19,hour=0, minute=0)
ref_time_1 = pd.Timestamp(year=2022, month=7, day=21,hour=0, minute=0)
ref_time_2 = pd.Timestamp(year=2022, month=7, day=22,hour=0, minute=0)
ref_time_3 = pd.Timestamp(year=2022, month=7, day=23,hour=0, minute=0)
ref_time_4 = pd.Timestamp(year=2022, month=7, day=24,hour=0, minute=0)
ref_time_5 = pd.Timestamp(year=2022, month=7, day=25,hour=0, minute=0)



for i, ax in enumerate(axes[0, :]):

    # plot location data on 21 July
    location_history = locations.loc[(locations['TIME_SERVER'] >= ref_time_0) & (locations['TIME_SERVER'] <= ref_time_5)]
    lons             = location_history['LON_DEC'].values
    lats             = location_history['LAT_DEC'].values
    times            = location_history['TIME_SERVER'].values

    ax.plot(lons,lats,color='w',linewidth=3,transform=proj_pl,zorder=10, linestyle='-')
    ax.plot(lons,lats,color='k',linewidth=2,transform=proj_pl,zorder=10, linestyle=':')
    # plot location data on 22 July
    location_history = locations.loc[(locations['TIME_SERVER'] >= ref_time_2) & (locations['TIME_SERVER'] <= ref_time_3)]
    lons             = location_history['LON_DEC'].values
    lats             = location_history['LAT_DEC'].values
    times            = location_history['TIME_SERVER'].values
    ax.plot(lons,lats,color='k',linewidth=3,transform=proj_pl,zorder=10, linestyle='-')
    ax.plot(lons,lats,color='white',linewidth=2,transform=proj_pl,zorder=10, linestyle='-')

for i, ax in enumerate(axes[1, :]):
    # plot historic location data 
    location_history = locations.loc[(locations['TIME_SERVER'] >= ref_time_0) & (locations['TIME_SERVER'] <= ref_time_5)]
    lons             = location_history['LON_DEC'].values
    lats             = location_history['LAT_DEC'].values
    times            = location_history['TIME_SERVER'].values
    ax.plot(lons,lats,color='w',linewidth=3,transform=proj_pl,zorder=10, linestyle='-')
    ax.plot(lons,lats,color='k',linewidth=2,transform=proj_pl,zorder=10, linestyle=':')
    # plot location data on 23 July
    location_history = locations.loc[(locations['TIME_SERVER'] >= ref_time_3) & (locations['TIME_SERVER'] <= ref_time_4)]
    lons             = location_history['LON_DEC'].values
    lats             = location_history['LAT_DEC'].values
    times            = location_history['TIME_SERVER'].values
    ax.plot(lons,lats,color='k',linewidth=3,transform=proj_pl,zorder=10, linestyle='-')
    ax.plot(lons,lats,color='white',linewidth=2,transform=proj_pl,zorder=10, linestyle='-')

annotations = [
    "22 Jul 2022 15:10",
    "23 Jul 2022 00:10",
    "23 Jul 2022 18:20",
    "23 Jul 2022 23:40",
]

image_locations  = pd.read_csv('/home/waynedj/Data/SCALE/data/track/SCALE-WIN22-SDS_1min.csv', parse_dates=['TIME_SERVER'])
annotation_times = pd.to_datetime(annotations, format="%d %b %Y %H:%M")
subset           = image_locations[image_locations["TIME_SERVER"].isin(annotation_times)]

for row in subset.itertuples():
    lon = row.LON_DEC
    lat = row.LAT_DEC
    time = row.TIME_SERVER
    for ax in axes.flat:
        if time <= pd.Timestamp(year=2022, month=7, day=23, hour=0, minute=0) and ax in axes[0, :]:
            ax.plot(lon, lat, marker='o', color='k', markersize=12, transform=proj_pl, zorder=11)
            ax.plot(lon, lat, marker='o', color='w', markersize=9, transform=proj_pl, zorder=11)

        if time <= pd.Timestamp(year=2022, month=7, day=23, hour=0, minute=0) and ax in axes[1, :]:
            ax.plot(lon, lat, marker='o', color='w', markersize=12, transform=proj_pl, zorder=11)
            ax.plot(lon, lat, marker='o', color='k', markersize=9, transform=proj_pl, zorder=11)

        if time >= pd.Timestamp(year=2022, month=7, day=23, hour=0, minute=0) and ax in axes[1, :]:
            ax.plot(lon, lat, marker='o', color='k', markersize=12, transform=proj_pl, zorder=11)
            ax.plot(lon, lat, marker='o', color='w', markersize=9, transform=proj_pl, zorder=11)


plt.savefig(f'/home/waynedj/Projects/swath_based_framework/figures/publication/Figure01_v002.png', dpi=500, bbox_inches='tight')
plt.show()
plt.close(fig)


timeit_t1 = timeit.default_timer()

print('[dt] Processing time: ' + str(np.round((timeit_t1-timeit_t0),3))+ 's')


# %%
