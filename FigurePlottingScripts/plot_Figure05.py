

#%%#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib import colormaps
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import pandas as pd
import re
import pyresample as pr
from pyresample.geometry import GridDefinition


# Paths to your PNG images
image_files = [
    '/home/waynedj/Data/SCALE/data/images/OnBoardCameraLeft/showcase/22-07-22 15-00-00.jpg',
    '/home/waynedj/Data/SCALE/data/images/OnBoardCameraLeft/showcase/22-07-23 00-10-00.jpg',
    '/home/waynedj/Data/SCALE/data/images/OnBoardCameraLeft/showcase/22-07-23 14-09-53.jpg',
    '/home/waynedj/Data/SCALE/data/images/OnBoardCameraLeft/showcase/22-07-23 18-20-00.jpg',
    '/home/waynedj/Data/SCALE/data/images/OnBoardCameraLeft/showcase/22-07-23 23-45-00.jpg',
]

BS_swath_files = [
    '/home/waynedj/Data/sic/jaxa_L2SIC/scale2022/GW1AM2_202207221506_087A_L2SGSICLC3300300.h5',
    '/home/waynedj/Data/sic/jaxa_L2SIC/scale2022/GW1AM2_202207230010_183D_L2SGSICLC3300300.h5',
    '/home/waynedj/Data/sic/jaxa_L2SIC/scale2022/GW1AM2_202207231410_078A_L2SGSICLC3300300.h5',
    '/home/waynedj/Data/sic/jaxa_L2SIC/scale2022/GW1AM2_202207231410_078A_L2SGSICLC3300300.h5',
    '/home/waynedj/Data/sic/jaxa_L2SIC/scale2022/GW1AM2_202207232314_174D_L2SGSICLC3300300.h5',
]

ECICE_swath_files = [
    '/home/waynedj/Data/intermediate/ecice/scale2022/res10/netcdf/GW1AM2_202207221506_087A_L1SGRTBR_2220220-ECICE.nc',
    '/home/waynedj/Data/intermediate/ecice/scale2022/res10/netcdf/GW1AM2_202207230010_183D_L1SGRTBR_2220220-ECICE.nc',
    '/home/waynedj/Data/intermediate/ecice/scale2022/res10/netcdf/GW1AM2_202207231410_078A_L1SGRTBR_2220220-ECICE.nc',
    '/home/waynedj/Data/intermediate/ecice/scale2022/res10/netcdf/GW1AM2_202207231410_078A_L1SGRTBR_2220220-ECICE.nc',
    '/home/waynedj/Data/intermediate/ecice/scale2022/res10/netcdf/GW1AM2_202207232314_174D_L1SGRTBR_2220220-ECICE.nc',
]

# Corresponding date–time annotations
annotations = [
    "22 Jul 2022 15:10",
    "23 Jul 2022 00:10",
    "23 Jul 2022 14:09",
    "23 Jul 2022 18:20",
    "23 Jul 2022 23:40",
]
annotation_times = pd.to_datetime(annotations, format="%d %b %Y %H:%M")

# setup parameters
grid_proj        = 'PS'
grid_file_12500  = "/home/waynedj/Data/grids/NSIDC0771_LatLon_PS_S12.5km_v1.1.nc"
ti_var_name      = "TI"
yi_var_name      = "YI"
fyi_var_name     = "FYI"
vmin             = 0
vmax             = 100
cmap             = "jet"
pcm_shading      = "auto"
extent           = [-120000, 90000, 3450000, 3600000]
proj_ease        = ccrs.LambertAzimuthalEqualArea(central_latitude=-90,central_longitude=0)
proj_ps          = ccrs.SouthPolarStereo(central_longitude=0)
proj_pl          = ccrs.PlateCarree()

track_density    = 30  # in minutes (1-min data -> 3-hour spacing)
locations        = pd.read_csv("/home/waynedj/Data/SCALE/data/track/SCALE-WIN22-SDS_1min.csv",parse_dates=["TIME_SERVER"],)
locations        = locations.iloc[::track_density, :].reset_index(drop=True)
lat_col          = "LAT_DEC"
lon_col          = "LON_DEC"

'''
def swath_remapping(in_lon, in_lat, in_data, out_grid, sample_radius):

    roi_m = sample_radius * 1000 #units km to meters
    # input grid: input swath data
    in_grid  = pr.geometry.SwathDefinition(
        lons=pr.utils.wrap_longitudes(in_lon),
        lats=in_lat,
    )
    # remap input_grid onto output_grid
    in_data  = np.asarray(in_data, dtype=np.float32)

    grdata   = pr.kd_tree.resample_nearest(in_grid, in_data, out_grid, radius_of_influence=roi_m, fill_value=np.nan)
    
    return grdata
'''

def custom_colormap(name):
    boundaries = np.arange(0, 100, 5)
    base_cmap = colormaps[name].resampled(len(boundaries))
    colors = base_cmap(np.arange(len(boundaries)))
    cmap_custom = mpl.colors.ListedColormap(colors, name="k")
    cmap_custom.set_over(colors[-1])
    return cmap_custom

def squeeze_first(da):
    for dim in ("time", "TIME", "t"):
        if dim in da.dims:
            return da.isel({dim: 0}).squeeze()
    return da.squeeze()

def swath_time_from_path(path):
    match = re.search(r"_(\d{12})_", path)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M")

def plot_track(ax, track_subset, lon_col, lat_col, marker_color="w"):
    lons = track_subset[lon_col].values
    lats = track_subset[lat_col].values
    if len(lons) == 0:
        return
    ax.plot(lons, lats, color="k", linewidth=3.5, transform=proj_pl, zorder=10)
    ax.plot(lons, lats, color="w", linewidth=2, transform=proj_pl, zorder=10)
    ax.plot(
        lons[-1],
        lats[-1],
        marker="o",
        color="k",
        markersize=12,
        transform=proj_pl,
        zorder=10,
    )
    ax.plot(
        lons[-1],
        lats[-1],
        marker="o",
        color=marker_color,
        markersize=9,
        transform=proj_pl,
        zorder=10,
    )


fig = plt.figure(figsize=(24, 19))
gs  = fig.add_gridspec(5, 5, wspace=0.01, hspace=0.01, bottom=0.17)

im_for_cbar = None

for i in range(5):
    is_bottom_row = (i == gs.nrows - 1)
    ax_img        = fig.add_subplot(gs[i, 0])
    ax_bs         = fig.add_subplot(gs[i, 1], projection=proj_ps)
    ax_ecice_ti   = fig.add_subplot(gs[i, 2], projection=proj_ps)
    ax_ecice_yi   = fig.add_subplot(gs[i, 3], projection=proj_ps)
    ax_ecice_fyi  = fig.add_subplot(gs[i, 4], projection=proj_ps)
    row_letter    = chr(ord("A") + i)

    if i == 0:
        ax_bs.set_title("Bootstrap", fontsize=20, pad=12)
        ax_ecice_ti.set_title(r"ECICE$_{total}$", fontsize=20, pad=12)
        ax_ecice_yi.set_title(r"ECICE$_{YI}$", fontsize=20, pad=12)
        ax_ecice_fyi.set_title(r"ECICE$_{FYI}$", fontsize=20, pad=12)

    img = mpimg.imread(image_files[i])
    ax_img.imshow(img, cmap="gray")
    ax_img.axis("off")

    # Text annotation (bottom-left corner)
    ax_img.text(0.02,0.03,annotation_times[i].strftime("%d %b %H:%M"),transform=ax_img.transAxes,ha="left",va="bottom",fontsize=14,color="white",bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),)
    ax_img.text(0.02,0.97,f"({row_letter}1)",transform=ax_img.transAxes,ha="left",va="top",fontsize=14,color="white",bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),)
    
    # bootstrap swath data remapped onto regular grid
    with xr.open_dataset(BS_swath_files[i]) as ds_bs:
        bs_data_raw = squeeze_first(ds_bs['Geophysical Data']).data/10
        bs_lon = ds_bs['Longitude of Observation Point'].data
        bs_lat = ds_bs['Latitude of Observation Point'].data

    im_bs = ax_bs.pcolormesh(bs_lon,bs_lat,bs_data_raw,cmap=custom_colormap(cmap),vmin=vmin,vmax=vmax,shading=pcm_shading,transform=proj_pl,)
    ax_bs.set_axis_off()
    ax_bs.set_extent(extent, crs=proj_ps)
    ax_bs.text(0.02,0.97,f"({row_letter}2)",transform=ax_bs.transAxes,ha="left",va="top",fontsize=14,color="white",bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),)
    gl = ax_bs.gridlines(draw_labels=True, linewidth=0.5, color="grey", alpha=1, linestyle="-")
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = is_bottom_row
    gl.xlabel_style = {"color": "k", "fontsize": 12}
    gl.ylabel_style = {"color": "k", "fontsize": 12}

    # ECICE swath data onto regular grid
    with xr.open_dataset(ECICE_swath_files[i]) as ds_ecice:
        ecice_ti_data = squeeze_first(ds_ecice[ti_var_name])
        ecice_yi_data = squeeze_first(ds_ecice[yi_var_name])
        ecice_fyi_data = squeeze_first(ds_ecice[fyi_var_name])
        ecice_lon = ds_ecice['longitude'].data
        ecice_lat = ds_ecice['latitude'].data

    im_ti = ax_ecice_ti.pcolormesh(ecice_lon,ecice_lat,ecice_ti_data,cmap=custom_colormap(cmap),vmin=vmin,vmax=vmax,shading=pcm_shading,transform=proj_pl,)
    im_yi = ax_ecice_yi.pcolormesh(ecice_lon,ecice_lat,ecice_yi_data,cmap=custom_colormap(cmap),vmin=vmin,vmax=vmax,shading=pcm_shading,transform=proj_pl,)
    im_fyi = ax_ecice_fyi.pcolormesh(ecice_lon,ecice_lat,ecice_fyi_data,cmap=custom_colormap(cmap),vmin=vmin,vmax=vmax,shading=pcm_shading,transform=proj_pl,)
    ax_ecice_ti.set_axis_off()
    ax_ecice_yi.set_axis_off()
    ax_ecice_fyi.set_axis_off()
    ax_ecice_ti.set_extent(extent, crs=proj_ps)
    ax_ecice_yi.set_extent(extent, crs=proj_ps)
    ax_ecice_fyi.set_extent(extent, crs=proj_ps)
    ax_ecice_ti.text(0.02,0.97,f"({row_letter}3)",transform=ax_ecice_ti.transAxes,ha="left",va="top",fontsize=14,color="white",bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),)
    ax_ecice_yi.text(0.02,0.97,f"({row_letter}4)",transform=ax_ecice_yi.transAxes,ha="left",va="top",fontsize=14,color="white",bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),)
    ax_ecice_fyi.text(0.02,0.97,f"({row_letter}5)",transform=ax_ecice_fyi.transAxes,ha="left",va="top",fontsize=14,color="white",bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),)
    for gl in [ax_ecice_ti, ax_ecice_yi, ax_ecice_fyi]:
        gl = gl.gridlines(draw_labels=True, linewidth=0.5, color="grey", alpha=1, linestyle="-")
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        gl.bottom_labels = is_bottom_row
        gl.xlabel_style = {"color": "k", "fontsize": 12}
        gl.ylabel_style = {"color": "k", "fontsize": 12}

    # Plot track up to the image timestamp
    track_subset = locations.loc[locations["TIME_SERVER"] <= annotation_times[i]]
    plot_track(ax_bs, track_subset, lon_col, lat_col, marker_color="w")
    plot_track(ax_ecice_ti, track_subset, lon_col, lat_col, marker_color="w")
    plot_track(ax_ecice_yi, track_subset, lon_col, lat_col, marker_color="w")
    plot_track(ax_ecice_fyi, track_subset, lon_col, lat_col, marker_color="w")

    # Annotate swath file time on second column (bottom-left)
    swath_time = swath_time_from_path(ECICE_swath_files[i])
    if swath_time is not None:
        ax_bs.text(
            0.02,
            0.03,
            swath_time.strftime("%d %b %H:%M"),
            transform=ax_bs.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
        )
    
    im_for_cbar = im_ti

# Add a shared colorbar at the bottom
left, bottom, width, height = 0.2, 0.10, 0.6, 0.03
new_width = 0.5
new_left = left - (new_width - width) / 2
cbar_ax = fig.add_axes([new_left, bottom, new_width, height])
cbar = fig.colorbar(im_for_cbar, cax=cbar_ax, orientation='horizontal', label=f'Sea Ice Concentration (%)')
cbar.ax.xaxis.label.set_size(20)
cbar.ax.tick_params(labelsize=18)

plt.savefig('/home/waynedj/Projects/swath_based_framework/figures/publication/Figure05_v001.png', dpi=800, bbox_inches='tight')
plt.show()
plt.close()

# %%
