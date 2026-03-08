#%%

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib as mpl
from matplotlib import colormaps
from matplotlib.path import Path as MplPath

def _plot_antimeridian_safe(ax, lons, lats, transform, linecolor, linestyle, linewidth, zorder_outline):
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    good = np.isfinite(lons) & np.isfinite(lats)
    if np.count_nonzero(good) < 2:
        return

    lons = lons[good]
    lats = lats[good]

    # split where lon jumps by > 180 deg
    split_idx = np.where(np.abs(np.diff(lons)) > 180.0)[0] + 1
    lon_parts = np.split(lons, split_idx)
    lat_parts = np.split(lats, split_idx)

    for lon_part, lat_part in zip(lon_parts, lat_parts):
        if lon_part.size >= 2:
            ax.plot(
                lon_part, lat_part,
                linestyle=linestyle, color=linecolor, linewidth=linewidth,
                transform=transform, zorder=zorder_outline
            )

def plot_swath_outline(ax, swath_lats, swath_lons, transform, zorder_outline=5):
    swath_lats = np.asarray(swath_lats, dtype=float)
    swath_lons = np.asarray(swath_lons, dtype=float)

    edge_lines = [
        (swath_lons[0, :],  swath_lats[0, :]),   # top edge
        (swath_lons[:, 0],  swath_lats[:, 0]),   # left edge
        (swath_lons[:, -1], swath_lats[:, -1]),  # right edge
        (swath_lons[-1, :], swath_lats[-1, :]),  # bottom edge
    ]

    for lons, lats in edge_lines:
        _plot_antimeridian_safe(ax, lons, lats, transform, "k", "-",  1.8, zorder_outline)
        _plot_antimeridian_safe(ax, lons, lats, transform, "w", "--", 1.2, zorder_outline)

def CustomColorMap2(cmap_name):
    boundaries = np.arange(-20, 20, 1)
    base_cmap = colormaps[cmap_name].resampled(len(boundaries))
    colors = base_cmap(np.arange(len(boundaries)))
    colors[0:int(-1 + len(boundaries)/2)] = colors[1:int(len(boundaries)/2)]
    colors[int(1 + len(boundaries)/2):int(len(boundaries))] = colors[int(len(boundaries)/2):int(-1 + len(boundaries))]
    colors[int(-1 + len(boundaries)/2)] = [1, 1, 1, 1]
    colors[int(len(boundaries)/2)] = [1, 1, 1, 1]
    cmap2 = mpl.colors.ListedColormap(colors, "k")
    cmap2.set_over(colors[-1])
    return cmap2

# -----------------------------------------------------------------------------
# INPUTS 
# -----------------------------------------------------------------------------
DECOMP_NC = "/home/waynedj/Projects/swath_based_framework/Data/sic_decomposition_jaxa_ecice_TI_YI_FYI_MYI_on_driftgrid.nc"
FDIR_JAXA = "/home/waynedj/Data/sic/jaxa_L2SIC/CaseStudy2019/"
f_sic_t0  = f"{FDIR_JAXA}/GW1AM2_201907261709_107A_L2SGSICLC3300300.h5"
f_sic_t1  = f"{FDIR_JAXA}/GW1AM2_201907270213_203D_L2SGSICLC3300300.h5"
# -----------------------------------------------------------------------------
# Load decomposition dataset generated using "AdvectSIC_CourseScale_v007.py"
# -----------------------------------------------------------------------------
ds    = xr.open_dataset(DECOMP_NC)
lon2d = ds["lon"].values
lat2d = ds["lat"].values
# Variables (all in %)
fields = {
    "Bootstrap": (
        ds["obs_dSIC_jaxa"].values,
        ds["adv_dSIC_jaxa"].values,
        ds["resid_jaxa"].values,
    ),
    "ECICE$_{TI}$": (
        ds["obs_dSIC_ecice_TI"].values,
        ds["adv_dSIC_ecice_TI"].values,
        ds["resid_ecice_TI"].values,
    ),
    "ECICE$_{YI}$": (
        ds["obs_dSIC_ecice_YI"].values,
        ds["adv_dSIC_ecice_YI"].values,
        ds["resid_ecice_YI"].values,
    ),
}

fields = {
    "Bootstrap": (
        ds["obs_dSIC_jaxa"].values,
        ds["adv_dSIC_jaxa"].values,
        ds["resid_jaxa"].values,),
    "ECICE$_{TI}$": (
        ds["obs_dSIC_ecice_TI"].values,
        ds["adv_dSIC_ecice_TI"].values,
        ds["resid_ecice_TI"].values,),}
# -----------------------------------------------------------------------------
# Load swath outlines (t0 and t1) from JAXA (for overlay only)
# -----------------------------------------------------------------------------
ds_sic0 = xr.open_dataset(f_sic_t0, engine="h5netcdf", phony_dims="sort")
ds_sic1 = xr.open_dataset(f_sic_t1, engine="h5netcdf", phony_dims="sort")
lat0_sw = ds_sic0["Latitude of Observation Point"].data
lon0_sw = ds_sic0["Longitude of Observation Point"].data
lat1_sw = ds_sic1["Latitude of Observation Point"].data
lon1_sw = ds_sic1["Longitude of Observation Point"].data
lon0_sw = ((lon0_sw + 180) % 360) - 180
lon1_sw = ((lon1_sw + 180) % 360) - 180
# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------
cmap = CustomColorMap2("seismic")
cmap.set_bad(color="0.8")  # grey NaNs

vmin = -20
vmax = 20
pcm_shading = "auto"

# CASE 1: extent = [-3000000, -500000, 500000, 3400000]
# CASE 2: extent = [-/, -/, /, /]
extent = [-3000000, -500000, 500000, 3400000]  # in SouthPolarStereo meters
proj_ps = ccrs.SouthPolarStereo(central_longitude=0)
proj_pl = ccrs.PlateCarree()



fig          = plt.figure(figsize=(16, 12))
gs           = fig.add_gridspec(2, 3, wspace=0.02, hspace=0.01, bottom=0.12, top=0.94)
axes         = np.empty((3, 3), dtype=object)
im           = [[None]*3 for _ in range(3)]
row_names    = list(fields.keys())
panel_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)' ]
col_titles   = ["ΔC$_{obs}$ (%)", "ΔC$_{D}$ (%)", "ΔC$_{res}$ (%)"]

for r, row_name in enumerate(row_names):
    obs_dSIC, adv_dSIC, resid = fields[row_name]
    row_data                  = [obs_dSIC, adv_dSIC, resid]
    print()
    print(row_name)
    print('max')
    print(np.nanmax(obs_dSIC))
    print(np.nanmax(adv_dSIC))
    print(np.nanmax(resid))
    print('min')
    print(np.nanmin(obs_dSIC))
    print(np.nanmin(adv_dSIC))
    print(np.nanmin(resid))

    for c in range(3):
        ax = fig.add_subplot(gs[r, c], projection=proj_ps)
        axes[r, c] = ax

        im[r][c] = ax.pcolormesh(
            lon2d, lat2d,
            row_data[c],
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading=pcm_shading, transform=proj_pl
        )

        # Titles: column titles on top row only
        if r == 0:
            ax.set_title(col_titles[c], fontsize=20)

        # Row labels on first column y-axis
        if c == 0:
            ax.text(-0.07, 0.5, row_name,
                    transform=ax.transAxes,
                    rotation=90, va="center", ha="right",
                    fontsize=20,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
                    clip_on=False)

        # Swath outlines (same outline overlay for all panels)
        plot_swath_outline(ax, swath_lats=lat0_sw, swath_lons=lon0_sw, transform=proj_pl, zorder_outline=6)
        plot_swath_outline(ax, swath_lats=lat1_sw, swath_lons=lon1_sw, transform=proj_pl, zorder_outline=6)

        # Panel labels (a-i) in white square boxes, top-right
        ax.text(
            0.04, 0.97, panel_labels[r * 3 + c],
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=16,
            bbox=dict(facecolor="white", edgecolor="k"),
            zorder=20
        )

        # Map cosmetics
        ax.set_extent(extent, crs=proj_ps)
        ax.add_feature(cfeature.LAND, facecolor="k", zorder=2)

        gl               = ax.gridlines(draw_labels=True)
        gl.xlocator      = mticker.FixedLocator(np.arange(-90, 10, 5))
        gl.ylocator      = mticker.FixedLocator(np.arange(-90, -40, 5))
        gl.right_labels  = True
        gl.bottom_labels = False
        gl.top_labels    = (r == 0)      # only top row shows top labels
        gl.left_labels   = (c == 0)     # only firs column shows left labels
        gl.xlabel_style  = {"size": 10}
        gl.ylabel_style  = {"size": 10}

# One shared colorbar for all subplots (bottom-center)
cax = fig.add_axes([0.2, 0.06, 0.6, 0.025])  # [left, bottom, width, height]
cb = fig.colorbar(
    im[0][0],
    cax=cax,
    orientation="horizontal"
)
cb.set_label("ΔSIC (%)", fontsize=20)
cb.ax.tick_params(labelsize=16)

plt.show()
#plt.savefig('/home/waynedj/Projects/swath_based_framework/figures/publication/Figure08_v001.png', dpi=500, bbox_inches='tight')
#plt.close()
#%%
