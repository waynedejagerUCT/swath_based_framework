#%%
import os
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colormaps
from matplotlib.gridspec import GridSpec
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
import re
from matplotlib.ticker import MultipleLocator
from datetime import datetime


# --------------------------------------------------
# Config: drift distributions
# --------------------------------------------------
DRIFT_FDIR              = "/home/waynedj/Data/seaicedrift/osisaf/swath/2019/07/"
DRIFT_N_SAMPLE          = 20
DRIFT_USE_RANDOM_SUBSET = False
LAT_RANGE               = [-75, -55]
LON_RANGE               = [-60, 0]
FONTSIZE                = 18
FONTSIZE_TICKLABELS     = 13
kde_on                  = False
_TS_RE                  = re.compile(r"_(\d{14})w_(\d{14})w(?:\.[^.]+)?$")
np.seterr(divide='ignore', invalid='ignore')
# --------------------------------------------------
# Config: TB/SIC distributions
# --------------------------------------------------
DIST_N_SAMPLE          = 10
DIST_USE_RANDOM_SUBSET = False
BS_SCALE_FACTOR        = 0.1
ICE_TYPE               = "TI"
RESAMPLE_RES           = "res10"
# --------------------------------------------------
# Paths
# --------------------------------------------------
F_TB_T0    = "/home/waynedj/Data/amsr2/l1r/CaseStudy2019/GW1AM2_201907261709_107A_L1SGRTBR_2220220.h5"
F_TB_T1    = "/home/waynedj/Data/amsr2/l1r/CaseStudy2019/GW1AM2_201907270213_203D_L1SGRTBR_2220220.h5"
F_BS_T0    = "/home/waynedj/Data/sic/jaxa_L2SIC/CaseStudy2019/GW1AM2_201907261709_107A_L2SGSICLC3300300.h5"
F_BS_T1    = "/home/waynedj/Data/sic/jaxa_L2SIC/CaseStudy2019/GW1AM2_201907270213_203D_L2SGSICLC3300300.h5"
F_ECICE_T0 = "/home/waynedj/Data/intermediate/ecice/CaseStudy2019/L1R/res10/netcdf/GW1AM2_201907261709_107A_L1SGRTBR_2220220-ECICE.nc"
F_ECICE_T1 = "/home/waynedj/Data/intermediate/ecice/CaseStudy2019/L1R/res10/netcdf/GW1AM2_201907270213_203D_L1SGRTBR_2220220-ECICE.nc"
FDIR_TB    = "/home/waynedj/Data/amsr2/l1r/CaseStudy2019/"
FDIR_BS    = "/home/waynedj/Data/sic/jaxa_L2SIC/CaseStudy2019/"
FDIR_ECICE = "/home/waynedj/Data/intermediate/ecice/CaseStudy2019/L1R/res10/netcdf/"


def custom_colormap(cmap_name):
    boundaries = np.arange(0, 500, 25)
    base_cmap = colormaps[cmap_name].resampled(len(boundaries))
    colors = base_cmap(np.arange(len(boundaries)))
    cmap = mpl.colors.ListedColormap(colors, name="k")
    cmap.set_over(colors[-1])
    return cmap

def build_drift_dataframe(ds):
    lat     = np.squeeze(ds["lat"].values)
    lon     = np.squeeze(ds["lon"].values)
    t0      = ds["t0"].values
    t1      = ds["t1"].values
    drift_x = np.squeeze(ds["driftX"].values)  # km
    drift_y = np.squeeze(ds["driftY"].values)  # km
    dt_sec  = (t1 - t0) / np.timedelta64(1, "s")
    u       = (drift_x * 1000.0) / dt_sec
    v       = (drift_y * 1000.0) / dt_sec
    u       = u.squeeze()
    v       = v.squeeze()
    dx, dy  = mpcalc.lat_lon_grid_deltas(lon * units.degree, lat * units.degree)
    div     = mpcalc.divergence(u * units("m/s"), v * units("m/s"), dx=dx, dy=dy).to("1/s").magnitude
    mask    = (
        (lat >= LAT_RANGE[0])
        & (lat <= LAT_RANGE[1])
        & (lon >= LON_RANGE[0])
        & (lon <= LON_RANGE[1])
        & np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(div))
    
    if not np.any(mask):
        return None

    return pd.DataFrame({"lat": lat[mask], "lon": lon[mask], "u": u[mask], "v": v[mask], "div": div[mask]})

def monthly_swath_files(base_dir, min_hours=5, max_hours=20):
    files = []
    for day in range(1, 32):
        day_dir = os.path.join(base_dir, f"{day:02d}")
        if not os.path.isdir(day_dir):
            continue

        for fname in os.listdir(day_dir):
            full_path = os.path.join(day_dir, fname)
            if not os.path.isfile(full_path):
                continue

            m = _TS_RE.search(fname)
            if not m:
                continue  # filename doesn't contain the expected timestamps

            t0_str, t1_str = m.group(1), m.group(2)
            try:
                t0 = datetime.strptime(t0_str, "%Y%m%d%H%M%S")
                t1 = datetime.strptime(t1_str, "%Y%m%d%H%M%S")
            except ValueError:
                continue

            dt_hours = (t1 - t0).total_seconds() / 3600.0
            if min_hours <= dt_hours <= max_hours:
                files.append(full_path)
    return files

def collect_drift_dataframe(base_dir, sample_size, use_subset):
    files = monthly_swath_files(base_dir)
    if use_subset and len(files) > sample_size:
        files = random.sample(files, sample_size)

    dfs = []
    for file_path in files:
        ds = xr.open_dataset(file_path)
        df = build_drift_dataframe(ds)
        ds.close()
        if df is not None:
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["lat", "lon", "u", "v", "div"])

def build_tb_dataframe(ds, lat_bounds, lon_bounds, scale_factor):
    lats = ds["Latitude of Observation Point for 89A"].values.ravel()[::2]
    lons = ds["Longitude of Observation Point for 89A"].values.ravel()[::2]
    tb_19_h = ds[f"Brightness Temperature ({RESAMPLE_RES},18.7GHz,H)"].values.ravel() / scale_factor
    tb_19_v = ds[f"Brightness Temperature ({RESAMPLE_RES},18.7GHz,V)"].values.ravel() / scale_factor
    tb_37_h = ds[f"Brightness Temperature ({RESAMPLE_RES},36.5GHz,H)"].values.ravel() / scale_factor
    tb_37_v = ds[f"Brightness Temperature ({RESAMPLE_RES},36.5GHz,V)"].values.ravel() / scale_factor

    mask = (
        (lats >= lat_bounds[0])
        & (lats <= lat_bounds[1])
        & (lons >= lon_bounds[0])
        & (lons <= lon_bounds[1])
        & np.isfinite(lats)
        & np.isfinite(lons)
    )
    if not np.any(mask):
        return None

    return pd.DataFrame(
        {
            "lat": lats[mask],
            "lon": lons[mask],
            "tb_19_h": tb_19_h[mask],
            "tb_19_v": tb_19_v[mask],
            "tb_37_h": tb_37_h[mask],
            "tb_37_v": tb_37_v[mask],
        }
    )

def build_bs_dataframe(ds, lat_bounds, lon_bounds, scale_factor):
    lats = ds["Latitude of Observation Point"].values.ravel()
    lons = ds["Longitude of Observation Point"].values.ravel()
    sic = ds["Geophysical Data"].values.ravel().astype(float) * scale_factor

    mask = (
        (lats >= lat_bounds[0])
        & (lats <= lat_bounds[1])
        & (lons >= lon_bounds[0])
        & (lons <= lon_bounds[1])
        & np.isfinite(lats)
        & np.isfinite(lons)
        & np.isfinite(sic)
        & (sic >= 0)
        & (sic <= 100)
    )
    if not np.any(mask):
        return None

    return pd.DataFrame({"lat": lats[mask], "lon": lons[mask], "sic": sic[mask]})

def build_ecice_dataframe(ds, lat_bounds, lon_bounds, scale_factor):
    lats = ds["latitude"].values.ravel()
    lons = ds["longitude"].values.ravel()
    sic = ds[ICE_TYPE].values.ravel().astype(float) * scale_factor

    mask = (
        (lats >= lat_bounds[0])
        & (lats <= lat_bounds[1])
        & (lons >= lon_bounds[0])
        & (lons <= lon_bounds[1])
        & np.isfinite(lats)
        & np.isfinite(lons)
        & np.isfinite(sic)
        & (sic >= 0)
        & (sic <= 100)
    )
    if not np.any(mask):
        return None

    return pd.DataFrame({"lat": lats[mask], "lon": lons[mask], "sic": sic[mask]})

def collect_dataframe(
    data_dir,
    builder,
    lat_bounds,
    lon_bounds,
    sample_size,
    use_subset,
    scale_factor,
    empty_columns,
    open_kwargs=None,
):
    files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, fname))]
    if use_subset and len(files) > sample_size:
        files = random.sample(files, sample_size)

    open_kwargs = open_kwargs or {}
    dfs = []
    for file_path in files:
        ds = xr.open_dataset(file_path, **open_kwargs)
        df = builder(ds, lat_bounds, lon_bounds, scale_factor)
        ds.close()
        if df is not None:
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=empty_columns)

def load_case_dataframe(file_path, builder, lat_bounds, lon_bounds, scale_factor, open_kwargs=None):
    open_kwargs = open_kwargs or {}
    ds = xr.open_dataset(file_path, **open_kwargs)
    df = builder(ds, lat_bounds, lon_bounds, scale_factor)
    ds.close()
    return df

def plot_hist_with_kde(
    ax,
    values,
    bins,
    hist_color,
    kde_color,
    label,
    alpha=0.35,
    linestyle="-",
    plot_hist=True,
    plot_curve=True,
    histtype="bar",
    linewidth=1,
):
    values = pd.Series(values).dropna().to_numpy()
    if values.size < 2:
        return

    if histtype == 'step':
        alpha = 1

    if plot_hist:
        ax.hist(
            values,
            bins=bins,
            density=True,
            color=hist_color,
            alpha=alpha,
            histtype=histtype,
            linewidth=linewidth,
            edgecolor=hist_color if histtype == "step" else "none",
            label=label,
        )

    if plot_curve:
        kde = gaussian_kde(values)
        x_grid = np.linspace(values.min(), values.max(), 300)
        ax.plot(x_grid, kde(x_grid), color=kde_color, linewidth=1.5, linestyle=linestyle, label=label)

def filter_nonzero_sic(values):
    vals = pd.Series(values).dropna().to_numpy(dtype=float)
    return vals[(vals > 0) & (vals <= 100)]


# --------------------------------------------------
# Figure and axes
# --------------------------------------------------
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor("white")

outer      = GridSpec(2, 2, height_ratios=[5, 5], width_ratios=[5, 5], hspace=0.12, wspace=0.15)
ax_div     = fig.add_subplot(outer[1, 0])

tb_inner   = outer[0, 0].subgridspec(2, 2, hspace=0.05, wspace=0.05)
ax_tb19h   = fig.add_subplot(tb_inner[0, 0])
ax_tb19v   = fig.add_subplot(tb_inner[0, 1])
ax_tb37h   = fig.add_subplot(tb_inner[1, 0])
ax_tb37v   = fig.add_subplot(tb_inner[1, 1])
tb_axes = {
    "tb_19_h": ax_tb19h,
    "tb_19_v": ax_tb19v,
    "tb_37_h": ax_tb37h,
    "tb_37_v": ax_tb37v,
}
ax_tb_shared = fig.add_subplot(outer[0, 0], frameon=False)
ax_tb_shared.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
for spine in ax_tb_shared.spines.values():
    spine.set_visible(False)

sic_inner     = outer[0, 1].subgridspec(2, 1, hspace=0.05)
ax_sic_bs     = fig.add_subplot(sic_inner[0, 0])
ax_sic_ecice  = fig.add_subplot(sic_inner[1, 0], sharex=ax_sic_bs)
ax_sic_shared = fig.add_subplot(outer[0, 1], frameon=False)
ax_sic_shared.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
for spine in ax_sic_shared.spines.values():
    spine.set_visible(False)

inner      = outer[1, 1].subgridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.05, wspace=0.05)
ax_histx   = fig.add_subplot(inner[0, 0])
ax_scatter = fig.add_subplot(inner[1, 0], sharex=ax_histx)
ax_histy   = fig.add_subplot(inner[1, 1], sharey=ax_scatter)

# --------------------------------------------------
# Drift data and plots
# --------------------------------------------------
df_main = collect_drift_dataframe(DRIFT_FDIR, DRIFT_N_SAMPLE, DRIFT_USE_RANDOM_SUBSET)

file_case01        = (DRIFT_FDIR + "27/icedrift_amsr2-gw1_tb37v-LapAtb37h-Lap_simplex_lev2_sh625_20190726165739w_20190727025045w.nc")
ds_case            = xr.open_dataset(file_case01)
df_case01          = build_drift_dataframe(ds_case)
ds_case.close()
x_main, y_main     = df_main["u"], df_main["v"]
x_case, y_case     = df_case01["u"], df_case01["v"]
div_main, div_case = df_main["div"], df_case01["div"]

bin_width   =   0.0000001
x_lim       = [-0.000002,0.000002]
y_lim       = [0, 1500000]
hist_bins   = np.arange(x_lim[0], x_lim[1] + bin_width, bin_width)

alpha = 0.4
ax_div.hist(div_main, bins=hist_bins, density=True, alpha=alpha, color="k", label='Subsample')
ax_div.hist(div_case, bins=hist_bins, density=True, alpha=alpha, color="darkorange",histtype='step', label='Case Study', linewidth=3)

div_main_clean = np.asarray(div_main[np.isfinite(div_main)])
div_case_clean = np.asarray(div_case[np.isfinite(div_case)])

if kde_on:
    if div_main_clean.size > 1 and div_case_clean.size > 1:
        kde_x = np.linspace(min(div_main_clean.min(), div_case_clean.min()), max(div_main_clean.max(), div_case_clean.max()), 400)
        ax_div.plot(kde_x, gaussian_kde(div_main_clean)(kde_x), color="darkorange", linewidth=1, label="Subsample")
        ax_div.plot(kde_x, gaussian_kde(div_case_clean)(kde_x), color="k", linewidth=1, label="Case Study")

ax_div.set_xlim(x_lim )
ax_div.set_ylim(y_lim)
ax_div.grid(True, alpha=0.5, color="k", linewidth=0.5, linestyle=":")
ax_div.set_ylabel("Probability density", fontsize=FONTSIZE, labelpad=15)
ax_div.set_xlabel("Divergence (s$^{-1}$)", fontsize=FONTSIZE, labelpad=12)
ax_div.legend(loc='upper right', fontsize=FONTSIZE - 4)

bin_width   =   0.02
x_lim       = [-0.5,0.5]
y_lim       = [-0.5,0.5]
hist2d_bins = np.arange(x_lim[0], x_lim[1] + bin_width, bin_width)
h2d, xedges, yedges = np.histogram2d(x_main, y_main, bins=hist2d_bins)
h2d                 = h2d.T
h2d[h2d == 0]       = np.nan

pcm = ax_scatter.pcolormesh(xedges, yedges, h2d, cmap=custom_colormap("Greys"), vmin=0, vmax=500)
ax_scatter.scatter(x_case, y_case, s=50, c="w", alpha=1, marker=".")
ax_scatter.scatter(x_case, y_case, s=30, c="darkorange", alpha=1, marker=".")
cax = inset_axes(ax_scatter, width="40%", height="3%", loc="lower left", borderpad=2)
fig.colorbar(pcm, cax=cax, orientation="horizontal")

alpha = 0.4
ax_histx.hist(x_main, bins=hist2d_bins, density=True, alpha=alpha, color="k")
ax_histx.hist(x_case, bins=hist2d_bins, density=True, alpha=alpha, color="darkorange", histtype='step', linewidth=3)
ax_histy.hist(y_main, bins=hist2d_bins, density=True, alpha=alpha, color="k", orientation="horizontal")
ax_histy.hist(y_case, bins=hist2d_bins, density=True, alpha=alpha, color="darkorange", histtype='step', linewidth=3, orientation="horizontal")
ax_histx.set_ylim([0,7])
ax_histy.set_xlim([0,7])

x_main_clean = np.asarray(pd.Series(x_main).dropna())
x_case_clean = np.asarray(pd.Series(x_case).dropna())
y_main_clean = np.asarray(pd.Series(y_main).dropna())
y_case_clean = np.asarray(pd.Series(y_case).dropna())

if kde_on:
    if x_main_clean.size > 1:
        x_grid_main = np.linspace(x_main_clean.min(), x_main_clean.max(), 300)
        ax_histx.plot(x_grid_main, gaussian_kde(x_main_clean)(x_grid_main), color="darkorange", linewidth=2)
    if x_case_clean.size > 1:
        x_grid_case = np.linspace(x_case_clean.min(), x_case_clean.max(), 300)
        ax_histx.plot(x_grid_case, gaussian_kde(x_case_clean)(x_grid_case), color="k", linewidth=2)

    if y_main_clean.size > 1:
        y_grid_main = np.linspace(y_main_clean.min(), y_main_clean.max(), 300)
        ax_histy.plot(gaussian_kde(y_main_clean)(y_grid_main), y_grid_main, color="darkorange", linewidth=2)
    if y_case_clean.size > 1:
        y_grid_case = np.linspace(y_case_clean.min(), y_case_clean.max(), 300)
        ax_histy.plot(gaussian_kde(y_case_clean)(y_grid_case), y_grid_case, color="k", linewidth=2)

ax_histx.tick_params(labelbottom=False)
ax_histy.tick_params(labelleft=False)

ax_histx.xaxis.set_major_locator(MultipleLocator(0.1))
ax_histx.yaxis.set_major_locator(MultipleLocator(2))
ax_histy.xaxis.set_major_locator(MultipleLocator(2))
ax_histy.yaxis.set_major_locator(MultipleLocator(0.1))

ax_histx.grid(True, linestyle=":", color="k", alpha=0.5)
ax_histy.grid(True, linestyle=":", color="k", alpha=0.5)

ax_scatter.set_xlabel(r"$u$-component (m.s$^{-1}$)", fontsize=FONTSIZE, labelpad=12)
ax_scatter.set_ylabel(r"$v$-component (m.s$^{-1}$)", fontsize=FONTSIZE, labelpad=10)
ax_scatter.grid(True, alpha=0.5, color="k", linewidth=0.5, linestyle=":")
ax_scatter.axhline(0, color="k", linewidth=1)
ax_scatter.axvline(0, color="k", linewidth=1)
ax_scatter.set_xlim([-0.5, 0.5])
ax_scatter.set_ylim([-0.5, 0.5])

# --------------------------------------------------
# TB/SIC data and plots
# --------------------------------------------------
h5_kwargs = {"engine": "h5netcdf", "phony_dims": "sort"}

df_main_tb = collect_dataframe(
    data_dir=FDIR_TB,
    builder=build_tb_dataframe,
    lat_bounds=LAT_RANGE,
    lon_bounds=LON_RANGE,
    sample_size=DIST_N_SAMPLE,
    use_subset=DIST_USE_RANDOM_SUBSET,
    scale_factor=100,
    empty_columns=["lat", "lon", "tb_19_h", "tb_19_v", "tb_37_h", "tb_37_v"],
    open_kwargs=h5_kwargs,
)
df_main_bs = collect_dataframe(
    data_dir=FDIR_BS,
    builder=build_bs_dataframe,
    lat_bounds=LAT_RANGE,
    lon_bounds=LON_RANGE,
    sample_size=DIST_N_SAMPLE,
    use_subset=DIST_USE_RANDOM_SUBSET,
    scale_factor=BS_SCALE_FACTOR,
    empty_columns=["lat", "lon", "sic"],
    open_kwargs=h5_kwargs,
)
df_main_ecice = collect_dataframe(
    data_dir=FDIR_ECICE,
    builder=build_ecice_dataframe,
    lat_bounds=LAT_RANGE,
    lon_bounds=LON_RANGE,
    sample_size=DIST_N_SAMPLE,
    use_subset=DIST_USE_RANDOM_SUBSET,
    scale_factor=1,
    empty_columns=["lat", "lon", "sic"],
)

df_case_tb_t0    = load_case_dataframe(F_TB_T0, build_tb_dataframe, LAT_RANGE, LON_RANGE, 100, open_kwargs=h5_kwargs)
df_case_tb_t1    = load_case_dataframe(F_TB_T1, build_tb_dataframe, LAT_RANGE, LON_RANGE, 100, open_kwargs=h5_kwargs)
df_case_bs_t0    = load_case_dataframe(F_BS_T0, build_bs_dataframe, LAT_RANGE, LON_RANGE, BS_SCALE_FACTOR, open_kwargs=h5_kwargs)
df_case_bs_t1    = load_case_dataframe(F_BS_T1, build_bs_dataframe, LAT_RANGE, LON_RANGE, BS_SCALE_FACTOR, open_kwargs=h5_kwargs)
df_case_ecice_t0 = load_case_dataframe(F_ECICE_T0, build_ecice_dataframe, LAT_RANGE, LON_RANGE, 1)
df_case_ecice_t1 = load_case_dataframe(F_ECICE_T1, build_ecice_dataframe, LAT_RANGE, LON_RANGE, 1)

bin_width   =  5
x_lim       = [90,275]
tb_bins     = np.arange(x_lim[0], x_lim[1] + bin_width, bin_width)
tb_alpha    = 0.4

tb_specs = [
    ("tb_19_h", 'limegreen', 'magenta', 'k', "Tb$_{{19H}}$"),
    ("tb_19_v", "limegreen", 'magenta', "k", "Tb$_{{19V}}$"),
    ("tb_37_h", "limegreen", 'magenta', "k", "Tb$_{{37H}}$"),
    ("tb_37_v", "limegreen", 'magenta', "k", "Tb$_{{37V}}$"),
]
for column, hist_color1, hist_color2, kde_color, base_label in tb_specs:
    ax_tb = tb_axes[column]
    plot_hist_with_kde(
        ax_tb,
        df_main_tb[column],
        tb_bins,
        'k',
        kde_color,
        f"{base_label}",
        alpha=tb_alpha,
        plot_hist=True,
        plot_curve=False,
    )
    plot_hist_with_kde(
        ax_tb,
        df_case_tb_t0[column],
        tb_bins,
        hist_color1,
        kde_color,
        f"{base_label} t$_{{0}}$",
        alpha=tb_alpha,
        linestyle="-",
        plot_hist=True,
        plot_curve=False,
        histtype="step",
        linewidth=2,
    )
    plot_hist_with_kde(
        ax_tb,
        df_case_tb_t1[column],
        tb_bins,
        hist_color2,
        kde_color,
        f"{base_label} t$_{{1}}$",
        alpha=tb_alpha,
        linestyle=":",
        plot_hist=True,
        plot_curve=False,
        histtype="step",
        linewidth=2,
    )

for column, _, _, _, _ in tb_specs:
    ax_tb = tb_axes[column]
    ax_tb.grid(True, linestyle=":", color="k", alpha=0.5)
    ax_tb.legend(loc='upper center', fontsize=FONTSIZE - 6)
    ax_tb.set_xlim([90, 275])
    ax_tb.set_ylim([0, 0.065])

ax_tb19h.tick_params(axis="x", labelbottom=False)
ax_tb19v.tick_params(axis="x", labelbottom=False)
ax_tb19v.tick_params(axis="y", labelleft=False)
ax_tb37v.tick_params(axis="y", labelleft=False)

ax_tb_shared.set_xlabel("Brightness Tempearture (k)", fontsize=FONTSIZE, labelpad=10)
ax_tb_shared.set_ylabel("Probability density", fontsize=FONTSIZE, labelpad=12)

sic_bins = np.arange(0, 101, 1)
sic_alpha = 0.4

for ax_sic, main_vals, case_t0_vals, case_t1_vals, color, base_label in [
    (ax_sic_bs, df_main_bs["sic"], df_case_bs_t0["sic"], df_case_bs_t1["sic"], "r", "Bootstrap"),
    (ax_sic_ecice, df_main_ecice["sic"], df_case_ecice_t0["sic"], df_case_ecice_t1["sic"], "b", "ECICE"),
]:
    main_vals_clean = filter_nonzero_sic(main_vals)
    case_t0_clean = filter_nonzero_sic(case_t0_vals)
    case_t1_clean = filter_nonzero_sic(case_t1_vals)

    plot_hist_with_kde(
        ax_sic,
        main_vals_clean,
        sic_bins,
        'k',
        color,
        f"{base_label}",
        alpha=sic_alpha,
        plot_hist=True,
        plot_curve=False
    )
    plot_hist_with_kde(
        ax_sic,
        case_t0_clean,
        sic_bins,
        'limegreen',
        color,
        rf"{base_label} t$_{{0}}$",
        alpha=sic_alpha,
        linestyle="-",
        plot_hist=True,
        plot_curve=False,
        histtype="step",
        linewidth=1.5,
    )
    plot_hist_with_kde(
        ax_sic,
        case_t1_clean,
        sic_bins,
        'magenta',
        color,
        rf"{base_label} t$_{{1}}$",
        alpha=sic_alpha,
        linestyle=":",
        plot_hist=True,
        plot_curve=False,
        histtype="step",
        linewidth=1.5,
    )
    ax_sic.grid(True, linestyle=":", color="k", alpha=0.5)
    ax_sic.legend(loc='upper center', fontsize=FONTSIZE - 4)
    ax_sic.set_xlim([80,100])
    ax_sic.set_ylim([0, 0.8])

ax_sic_bs.tick_params(axis="x", labelbottom=False)
ax_sic_ecice.set_xlabel("SIC (%)", fontsize=FONTSIZE, labelpad=10)
ax_sic_shared.set_ylabel("Probability density", fontsize=FONTSIZE, labelpad=12)

subplot_labels = [
    (ax_tb19h,     "(a)", 0.04, 0.96),
    (ax_tb19v,     "(b)", 0.04, 0.96),
    (ax_tb37h,     "(c)", 0.04, 0.96),
    (ax_tb37v,     "(d)", 0.04, 0.96),
    (ax_sic_bs,    "(e)", 0.02, 0.96),
    (ax_sic_ecice, "(f)", 0.02, 0.96),
    (ax_div,       "(g)", 0.02, 0.98),
    (ax_scatter,   "(h)", 0.02, 0.98),
    (ax_histx,     "(i)", 0.02, 0.94),
    (ax_histy,     "(j)", 0.08, 0.98),
]
for ax, label, x_loc, y_loc in subplot_labels:
    ax.text(
        x_loc,
        y_loc,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONTSIZE,
        color="black",
        bbox=dict(
            boxstyle="square,pad=0.2",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
        ),
    )

# Unified tick label font size across all subplot axes
for ax in [ax_tb19h, ax_tb19v, ax_tb37h, ax_tb37v, ax_sic_bs, ax_sic_ecice, ax_div, ax_scatter, ax_histx, ax_histy]:
    ax.tick_params(axis="both", which="both", labelsize=FONTSIZE_TICKLABELS)


plt.savefig('/home/waynedj/Projects/swath_based_framework/figures/publication/Figure07_v001.png', dpi=500, bbox_inches='tight')
plt.close()
# %%
