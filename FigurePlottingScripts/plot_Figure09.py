#%%

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
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

    split_idx = np.where(np.abs(np.diff(lons)) > 180.0)[0] + 1
    lon_parts = np.split(lons, split_idx)
    lat_parts = np.split(lats, split_idx)

    for lon_part, lat_part in zip(lon_parts, lat_parts):
        if lon_part.size >= 2:
            ax.plot(
                lon_part,
                lat_part,
                linestyle=linestyle,
                color=linecolor,
                linewidth=linewidth,
                transform=transform,
                zorder=zorder_outline,
            )

def plot_swath_outline(ax, swath_lats, swath_lons, transform, zorder_outline=5):
    swath_lats = np.asarray(swath_lats, dtype=float)
    swath_lons = np.asarray(swath_lons, dtype=float)

    edge_lines = [
        (swath_lons[0, :], swath_lats[0, :]),
        (swath_lons[:, 0], swath_lats[:, 0]),
        (swath_lons[:, -1], swath_lats[:, -1]),
        (swath_lons[-1, :], swath_lats[-1, :]),
    ]

    for lons, lats in edge_lines:
        _plot_antimeridian_safe(ax, lons, lats, transform, "k", "-", 1.8, zorder_outline)
        _plot_antimeridian_safe(ax, lons, lats, transform, "w", "--", 1.2, zorder_outline)

def swath_polygon_path(swath_lats, swath_lons):
    lat = np.asarray(swath_lats, dtype=float)
    lon = np.asarray(swath_lons, dtype=float)

    top = np.column_stack((lon[0, :], lat[0, :]))
    right = np.column_stack((lon[1:, -1], lat[1:, -1]))
    bottom = np.column_stack((lon[-1, -2::-1], lat[-1, -2::-1]))
    left = np.column_stack((lon[-2:0:-1, 0], lat[-2:0:-1, 0]))
    poly = np.vstack((top, right, bottom, left))
    poly = poly[np.isfinite(poly).all(axis=1)]

    if poly.shape[0] < 3:
        return None

    return MplPath(poly)

def footprint_mask_on_grid(lat2d, lon2d, swath_lats, swath_lons):
    path = swath_polygon_path(swath_lats, swath_lons)
    if path is None:
        return np.zeros_like(lat2d, dtype=bool)

    pts = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    return path.contains_points(pts).reshape(lat2d.shape)

def CustomColorMap2(cmap_name):
    boundaries = np.arange(-40, 40, 2)
    base_cmap = colormaps[cmap_name].resampled(len(boundaries))
    colors = base_cmap(np.arange(len(boundaries)))
    colors[0 : int(-1 + len(boundaries) / 2)] = colors[1 : int(len(boundaries) / 2)]
    colors[int(1 + len(boundaries) / 2) : int(len(boundaries))] = colors[
        int(len(boundaries) / 2) : int(-1 + len(boundaries))
    ]
    colors[int(-1 + len(boundaries) / 2)] = [1, 1, 1, 1]
    colors[int(len(boundaries) / 2)] = [1, 1, 1, 1]
    cmap2 = mpl.colors.ListedColormap(colors, "k")
    cmap2.set_over(colors[-1])
    return cmap2

def extract_sic_with_ti_mask(path, ice_types):
    ds = xr.open_dataset(path)
    lats = ds["latitude"].values.ravel().astype(float)
    lons = ds["longitude"].values.ravel().astype(float)
    ti_vals = ds["TI"].values.ravel().astype(float)

    geo_mask = (
        np.isfinite(lats)
        & np.isfinite(lons)
        & (lats >= LAT_RANGE[0])
        & (lats <= LAT_RANGE[1])
        & (lons >= LON_RANGE[0])
        & (lons <= LON_RANGE[1])
    )
    ti_mask = np.isfinite(ti_vals) & (ti_vals > 0) & (ti_vals <= 100)
    base_mask = geo_mask & ti_mask

    out = {}
    for ice_type in ice_types:
        vals = ds[ice_type].values.ravel().astype(float)
        mask = base_mask & np.isfinite(vals) & (vals <= 100)
        out[ice_type] = vals[mask]

    ds.close()
    return out

def collect_all_sic_by_type(data_dir, ice_types):
    out = {k: [] for k in ice_types}
    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith("-ECICE.nc") and os.path.isfile(os.path.join(data_dir, f))
    )

    for path in files:
        sic_dict = extract_sic_with_ti_mask(path, ice_types)
        for k in ice_types:
            if sic_dict[k].size:
                out[k].append(sic_dict[k])

    for k in ice_types:
        if out[k]:
            out[k] = np.concatenate(out[k])
        else:
            out[k] = np.array([], dtype=float)

    return out


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
DECOMP_NC = "/home/waynedj/Projects/swath_based_framework/Data/sic_decomposition_jaxa_ecice_TI_YI_FYI_MYI_on_driftgrid.nc"
F_ECICE_T0 = "/home/waynedj/Data/intermediate/ecice/CaseStudy2019/L1R/res10/netcdf/GW1AM2_201907261709_107A_L1SGRTBR_2220220-ECICE.nc"
F_ECICE_T1 = "/home/waynedj/Data/intermediate/ecice/CaseStudy2019/L1R/res10/netcdf/GW1AM2_201907270213_203D_L1SGRTBR_2220220-ECICE.nc"
FDIR_ECICE = "/home/waynedj/Data/intermediate/ecice/CaseStudy2019/L1R/res10/netcdf/"
LAT_RANGE = [-75, -55]
LON_RANGE = [-60, 0]
FONTSIZE_TICKLABELS     = 13
# -----------------------------------------------------------------------------
# Load decomposition and swaths
# -----------------------------------------------------------------------------
ds = xr.open_dataset(DECOMP_NC)
lon2d = ds["lon"].values
lat2d = ds["lat"].values

obs_var_map = {
    "TI": "obs_dSIC_ecice_TI",
    "YI": "obs_dSIC_ecice_YI",
    "FYI": "obs_dSIC_ecice_FYI",
    "MYI": "obs_dSIC_ecice_MYI",
}
obs_fields = {}
for key, var in obs_var_map.items():
    if var in ds:
        obs_fields[key] = ds[var].values
    else:
        print(f"[plot_Figure09] Warning: '{var}' not found in dataset; skipping {key}.")

ds_t0   = xr.open_dataset(F_ECICE_T0)
ds_t1   = xr.open_dataset(F_ECICE_T1)
lat0_sw = ds_t0["latitude"].values
lon0_sw = ds_t0["longitude"].values
lat1_sw = ds_t1["latitude"].values
lon1_sw = ds_t1["longitude"].values
ds_t0.close()
ds_t1.close()

lon0_sw = ((lon0_sw + 180) % 360) - 180
lon1_sw = ((lon1_sw + 180) % 360) - 180

# -----------------------------------------------------------------------------
# Build footprint mask (union of t0 + t1) for grey outside area
# -----------------------------------------------------------------------------
mask_t0 = footprint_mask_on_grid(lat2d, lon2d, lat0_sw, lon0_sw)
mask_t1 = footprint_mask_on_grid(lat2d, lon2d, lat1_sw, lon1_sw)
mask_union = mask_t0 | mask_t1

obs_plot = {}
for k, arr in obs_fields.items():
    obs_plot[k] = np.where(mask_union, arr, np.nan)

def print_pcolormesh_minmax(label, arr):
    finite = np.isfinite(arr)
    if not np.any(finite):
        print(f"[plot_Figure09] {label} pcolormesh: no finite values after masking.")
        return
    vmin_plot = float(np.nanmin(arr))
    vmax_plot = float(np.nanmax(arr))
    print(f"[plot_Figure09] {label} pcolormesh: min={vmin_plot:.3f}, max={vmax_plot:.3f}")

for key in ["TI", "YI", "FYI", "MYI"]:
    if key in obs_plot:
        print_pcolormesh_minmax(key, obs_plot[key])
    else:
        print(f"[plot_Figure09] {key} pcolormesh: not available (missing source variable).")

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
cmap = CustomColorMap2("seismic")
cmap.set_bad(color="0.8")
vmin, vmax = -40, 40

extent = [-3000000, -500000, 500000, 3400000]
proj_ps = ccrs.SouthPolarStereo(central_longitude=0)
proj_pl = ccrs.PlateCarree()

fig = plt.figure(figsize=(14, 16))
outer = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.03, bottom=0.10, top=0.96)

ax_ti = fig.add_subplot(outer[0, 0], projection=proj_ps)
ax_yi = fig.add_subplot(outer[0, 1], projection=proj_ps)
ax_fyi = fig.add_subplot(outer[1, 0], projection=proj_ps)
ax_myi = fig.add_subplot(outer[1, 1], projection=proj_ps)

map_axes = [
    (ax_ti, "TI", "(a)"),
    (ax_yi, "YI", "(b)"),
    (ax_fyi, "FYI", "(c)"),
    (ax_myi, "MYI", "(d)"),
]

im = None
for ax, key, panel in map_axes:
    im = ax.pcolormesh(
        lon2d,
        lat2d,
        obs_plot[key],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        transform=proj_pl,
    )

    plot_swath_outline(ax, swath_lats=lat0_sw, swath_lons=lon0_sw, transform=proj_pl, zorder_outline=6)
    plot_swath_outline(ax, swath_lats=lat1_sw, swath_lons=lon1_sw, transform=proj_pl, zorder_outline=6)

    ax.text(
        0.02,
        0.015,
        f"ΔC$_{{obs}}$ {key} (%)",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=14,
        zorder=20,
    )
    ax.text(
        0.025,
        0.975,
        panel,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=18,
        bbox=dict(facecolor="white", edgecolor="k"),
        zorder=20,
    )

    ax.set_extent(extent, crs=proj_ps)
    ax.add_feature(cfeature.LAND, facecolor="k", zorder=2)

    gl = ax.gridlines(draw_labels=True)
    gl.xlocator = mticker.FixedLocator(np.arange(-90, 10, 5))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, -40, 5))
    gl.right_labels = True
    gl.bottom_labels = False
    #gl.left_labels = (ax in [ax_yi, ax_fyi])
    #gl.top_labels = (ax == ax_yi)
    gl.xlabel_style = {"size": FONTSIZE_TICKLABELS}
    gl.ylabel_style = {"size": FONTSIZE_TICKLABELS}

# Shared colorbar (bottom)
cax = fig.add_axes([0.22, 0.06, 0.56, 0.025])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("ΔSIC (%)", fontsize=18)
cb.ax.tick_params(labelsize=FONTSIZE_TICKLABELS)

# Unified tick label size for all subplot axes
for ax in [ax_ti, ax_yi, ax_fyi, ax_myi]:
    ax.tick_params(axis="both", which="both", labelsize=FONTSIZE_TICKLABELS-3)


plt.savefig("/home/waynedj/Projects/swath_based_framework/figures/publication/Figure09_v002.png",dpi=500,bbox_inches="tight",)
plt.close()
#%%
