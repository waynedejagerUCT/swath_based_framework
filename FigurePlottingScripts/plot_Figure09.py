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

def extract_ice_type_sic(path, ice_type):
    ds = xr.open_dataset(path)
    lats = ds["latitude"].values.ravel().astype(float)
    lons = ds["longitude"].values.ravel().astype(float)
    vals = ds[ice_type].values.ravel().astype(float)
    ds.close()
    mask = (
        np.isfinite(lats)
        & np.isfinite(lons)
        & (lats >= LAT_RANGE[0])
        & (lats <= LAT_RANGE[1])
        & (lons >= LON_RANGE[0])
        & (lons <= LON_RANGE[1])
        & np.isfinite(vals)
        & (vals > 0)
        & (vals <= 100)
    )
    return vals[mask]

def collect_all_sic_by_type(data_dir, ice_types):
    out = {k: [] for k in ice_types}
    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith("-ECICE.nc") and os.path.isfile(os.path.join(data_dir, f))
    )

    for path in files:
        ds = xr.open_dataset(path)
        lats = ds["latitude"].values.ravel().astype(float)
        lons = ds["longitude"].values.ravel().astype(float)
        geo_mask = (
            np.isfinite(lats)
            & np.isfinite(lons)
            & (lats >= LAT_RANGE[0])
            & (lats <= LAT_RANGE[1])
            & (lons >= LON_RANGE[0])
            & (lons <= LON_RANGE[1])
        )
        for k in ice_types:
            vals = ds[k].values.ravel().astype(float)
            vals = vals[geo_mask & np.isfinite(vals) & (vals > 0) & (vals <= 100)]
            if vals.size:
                out[k].append(vals)
        ds.close()

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

obs_fields = {
    "YI": ds["obs_dSIC_ecice_YI"].values,
    "FYI": ds["obs_dSIC_ecice_FYI"].values,
    "MYI": ds["obs_dSIC_ecice_MYI"].values,
}

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

# -----------------------------------------------------------------------------
# Histogram datasets
# -----------------------------------------------------------------------------
hist_types = ["TI", "YI", "FYI", "MYI"]

all_sic = collect_all_sic_by_type(FDIR_ECICE, hist_types)
t0_sic = {k: extract_ice_type_sic(F_ECICE_T0, k) for k in hist_types}
t1_sic = {k: extract_ice_type_sic(F_ECICE_T1, k) for k in hist_types}

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
outer = fig.add_gridspec(2, 2, wspace=0.13, hspace=0.08, bottom=0.10, top=0.96)

# Top-left: inner 2x2 histogram panel
inner       = outer[0, 0].subgridspec(2, 2, hspace=0.08, wspace=0.08)
ax_ti       = fig.add_subplot(inner[0, 0])
ax_yi_hist  = fig.add_subplot(inner[0, 1], sharex=ax_ti)
ax_fyi_hist = fig.add_subplot(inner[1, 0], sharex=ax_ti)
ax_myi_hist = fig.add_subplot(inner[1, 1], sharex=ax_ti)

hist_axes = {
    "TI": ax_ti,
    "YI": ax_yi_hist,
    "FYI": ax_fyi_hist,
    "MYI": ax_myi_hist,
}
hist_ylims = {
    "TI": [0,0.45],
    "YI": [0,0.04],
    "FYI": [0,0.2],
    "MYI": [0,0.04],
}

sic_bins = np.arange(0, 102, 2)
for k, ax in hist_axes.items():
    ax.hist(
        all_sic[k],
        bins=sic_bins,
        density=True,
        histtype="bar",
        linewidth=1.0,
        color="k",
        alpha=0.4,
        label=f"{k}",
    )
    ax.hist(
        t0_sic[k],
        bins=sic_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color="limegreen",
        alpha=0.8,
        label=f"{k} t$_{{0}}$",
    )
    ax.hist(
        t1_sic[k],
        bins=sic_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color="magenta",
        alpha=0.8,
        label=f"{k} t$_{{1}}$",
    )
    ax.set_xlim(0, 100)
    if hist_ylims.get(k) is not None:
        ax.set_ylim(hist_ylims[k])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, linestyle=":", color="k", alpha=0.45)
    ax.legend(loc="upper center", fontsize=8)

ax_ti.tick_params(axis="x", labelbottom=False)
ax_yi_hist.tick_params(axis="x", labelbottom=False)
ax_yi_hist.yaxis.set_ticks_position("right")
ax_yi_hist.tick_params(axis="y", labelleft=False, labelright=True)
ax_myi_hist.yaxis.set_ticks_position("right")
ax_myi_hist.tick_params(axis="y", labelleft=False, labelright=True)

# Shared y-label for inner histogram block
ax_hist_shared = fig.add_subplot(outer[0, 0], frameon=False)
ax_hist_shared.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
for spine in ax_hist_shared.spines.values():
    spine.set_visible(False)
ax_hist_shared.set_ylabel("Probability Density", fontsize=18, labelpad=22)
ax_hist_shared.set_xlabel("SIC (%)", fontsize=18)

# Inner panel labels
for ax, lab in [(ax_ti, "(a)"), (ax_yi_hist, "(b)"), (ax_fyi_hist, "(c)"), (ax_myi_hist, "(d)")]:
    ax.text(
        0.04,
        0.97,
        lab,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="k"),
        zorder=20,
    )

# Other 3 panels: dCobs maps
ax_yi = fig.add_subplot(outer[0, 1], projection=proj_ps)
ax_fyi = fig.add_subplot(outer[1, 0], projection=proj_ps)
ax_myi = fig.add_subplot(outer[1, 1], projection=proj_ps)

map_axes = [
    (ax_yi, "YI", "(e)"),
    (ax_fyi, "FYI", "(f)"),
    (ax_myi, "MYI", "(g)"),
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
    gl.left_labels = (ax in [ax_yi, ax_fyi])
    gl.top_labels = (ax == ax_yi)
    gl.xlabel_style = {"size": FONTSIZE_TICKLABELS}
    gl.ylabel_style = {"size": FONTSIZE_TICKLABELS}

# Shared colorbar (bottom)
cax = fig.add_axes([0.22, 0.06, 0.56, 0.025])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("ΔSIC (%)", fontsize=18)
cb.ax.tick_params(labelsize=FONTSIZE_TICKLABELS)

# Unified tick label size for all subplot axes
for ax in [ax_ti, ax_yi_hist, ax_fyi_hist, ax_myi_hist, ax_yi, ax_fyi, ax_myi]:
    ax.tick_params(axis="both", which="both", labelsize=FONTSIZE_TICKLABELS-3)


plt.savefig("/home/waynedj/Projects/swath_based_framework/figures/publication/Figure09_v001.png",dpi=500,bbox_inches="tight",)
plt.close()

#%%
