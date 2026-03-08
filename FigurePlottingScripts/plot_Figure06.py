
#%%
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
import xarray as xr
import matplotlib.ticker as mticker


def transform_latlon_to_xy_km(lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6932", always_xy=True)
    x_m, y_m = transformer.transform(lon, lat)  # lon,lat order
    return np.asarray(x_m) / 1000.0, np.asarray(y_m) / 1000.0

def colors_from_cmap(cmap_name, n):
    cmap = mpl.colormaps.get_cmap(cmap_name)
    if n <= 1:
        return [mpl.colors.to_hex(cmap(0.5), keep_alpha=False)]
    samples = [i / (n - 1) for i in range(n)]
    return [mpl.colors.to_hex(cmap(s), keep_alpha=False) for s in samples]

def distance_time_series_m(model_x_km, model_y_km, model_time, buoy_xy_km_df):
    """Legacy helper (kept for compatibility). Returns meters."""
    model_t = pd.DatetimeIndex(pd.to_datetime(model_time))
    buoy_on_model = buoy_xy_km_df.reindex(model_t, method="nearest", tolerance=pd.Timedelta("20min"))
    dx_m = (model_x_km - buoy_on_model["x_km"].to_numpy()) * 1000.0
    dy_m = (model_y_km - buoy_on_model["y_km"].to_numpy()) * 1000.0
    dist_m = np.sqrt(dx_m**2 + dy_m**2)
    return model_t, dist_m

def get_release_slice(release_id, Nens):
    """Return slice for trajectories belonging to a release group."""
    i0 = release_id * Nens
    i1 = (release_id + 1) * Nens
    return slice(i0, i1)

def ensemble_distance_stats(ds, traj_slice, buoy_xy_km_df):
    """
    Compute mean, p05, p95 distance-to-buoy time series across ensemble members
    for a given trajectory slice.
    Returns: t (DatetimeIndex), mean_km, p05_km, p95_km
    """
    # Extract arrays: shape (Nens, obs)
    x = ds["lon"].isel(trajectory=traj_slice).values
    y = ds["lat"].isel(trajectory=traj_slice).values
    t = ds["time"].isel(trajectory=traj_slice).values

    # Use time from first ensemble member (they should be identical)
    t0 = t[0, :]
    model_t = pd.DatetimeIndex(pd.to_datetime(t0))

    buoy_on_model = buoy_xy_km_df.reindex(model_t, method="nearest", tolerance=pd.Timedelta("20min"))
    bx = buoy_on_model["x_km"].to_numpy()[None, :]  # (1, obs)
    by = buoy_on_model["y_km"].to_numpy()[None, :]

    dx = x - bx
    dy = y - by
    dist_km = np.sqrt(dx**2 + dy**2)

    mean_km = np.nanmean(dist_km, axis=0)
    p05_km  = np.nanpercentile(dist_km, 5, axis=0)
    p95_km  = np.nanpercentile(dist_km, 95, axis=0)

    return model_t, mean_km, p05_km, p95_km

# -------------------------
# Setup
# -------------------------
dataset_version = "v004"
P_list          = ["P106", "P108"]

# Buoy reference timeline
t_ref_start      = pd.Timestamp("2019-07-26 12:00:00")
t_ref_end        = pd.Timestamp("2019-07-30 12:00:00")
buoy_times_10min = pd.date_range(t_ref_start, t_ref_end, freq="10min")

# Ensemble settings in trajectory files
Nens      = 100   # ensemble members per release
n_release = 4     # number of releases (groups)

# Which release groups to plot (0..3). Example: [0] plots the first 100 trajectories.
release_groups_to_plot = [0, 1, 2, 3]

# For map readability: plot every Nth ensemble member
map_thin = 10

# Colors per release group
group_colors_from_map = colors_from_cmap("viridis", max(n_release, 2))
group_colors = ['orange', 'blue', 'green', 'red']

# Map projection
proj_ps = ccrs.SouthPolarStereo(central_longitude=0)

# Figure: 2 rows x 2 cols
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, wspace=0.01, hspace=0.05)

ax_map = {
    "P108": fig.add_subplot(gs[0, 0], projection=proj_ps),
    "P106": fig.add_subplot(gs[1, 0], projection=proj_ps),
}
ax_ts = {
    "P108": fig.add_subplot(gs[0, 1]),
    "P106": fig.add_subplot(gs[1, 1]),
}
ax_extent = {
    "P108": [-2070000, -2010000, 2390000, 2440000],
    "P106": [-2120000, -2060000, 2310000, 2360000],
}
legend_labels = ['26 Jul 12:00', '27 Jul 12:00', '28 Jul 12:00', '29 Jul 12:00']
# Style time-series axes
for P in P_list:
    axt = ax_ts[P]
    axt.set_ylabel(f"Distance to {P} (km)", fontsize=14)
    axt.set_xlabel("Time (UTC)", fontsize=14)
    axt.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    axt.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    axt.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    axt.grid(True, which="both", linestyle="-", linewidth=0.4, color="k", alpha=0.5)
    plt.setp(axt.get_xticklabels(), rotation=-50, ha="left")

# Style map axes
for P in P_list:
    axm = ax_map[P]
    axm.set_extent(ax_extent[P], crs=proj_ps)
    gl = axm.gridlines(
        draw_labels=True,
        x_inline=False,
        y_inline=False,
        xlocs=mticker.FixedLocator(np.arange(-44, -36, 0.1)),
        ylocs=mticker.FixedLocator(np.arange(-66, -60, 0.1)),)
    gl.linewidth = 0.4
    gl.top_labels = False if P == "P106" else True
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = False if P == "P108" else True

# -------------------------
# Main
# -------------------------
for P in P_list:
    axm = ax_map[P]
    axt = ax_ts[P]

    # Fetch buoy data
    buoy_csv = glob.glob(f"/home/waynedj/Data/seaicedrift/AWIbuoys/2019{P}_*.csv")[0]
    dfP = pd.read_csv(buoy_csv, parse_dates=["time"]).sort_values("time")

    # Build buoy reference track at 10-min cadence (nearest obs in the CSV)
    buoy_lat, buoy_lon = [], []
    for t in buoy_times_10min:
        idx = (dfP["time"] - t).abs().idxmin()
        row = dfP.loc[idx]
        buoy_lat.append(row["latitude (deg)"])
        buoy_lon.append(row["longitude (deg)"])

    buoy_x_km, buoy_y_km = transform_latlon_to_xy_km(buoy_lat, buoy_lon)
    buoy_xy_km_df        = (pd.DataFrame({"x_km": buoy_x_km, "y_km": buoy_y_km}, index=pd.DatetimeIndex(buoy_times_10min)).sort_index())

    # Plot buoy on map
    axm.plot(buoy_x_km * 1000,buoy_y_km * 1000,color="k",linewidth=2,transform=proj_ps,label=f"{P}",linestyle="-",)

    # Open trajectory datasets
    dsDM = xr.open_zarr(
        f"/home/waynedj/Projects/swath_based_framework/Data/TrajectoryData_CaseStudy2019_Daily_{P}_{dataset_version}.zarr"
    )
    dsS2S = xr.open_zarr(
        f"/home/waynedj/Projects/swath_based_framework/Data/TrajectoryData_CaseStudy2019_Swath_{P}_{dataset_version}.zarr"
    )

    for g in release_groups_to_plot:
        if g < 0 or g >= n_release:
            continue

        c          = group_colors[g]
        traj_slice = get_release_slice(g, Nens)
        kposDM  = dsDM.trajectory[0].values 
        kposS2S = dsS2S.trajectory[0].values
        # -------------------------
        # Map: plot a thinned subset of ensemble tracks for DM and S2S
        # -------------------------
        member_ids = np.arange(Nens)[::map_thin]  # thin for readability
        for m_id in member_ids:
            k       =  g * Nens + m_id
            # DM member track
            dm_x_km = dsDM["lon"].isel(trajectory=k).values
            dm_y_km = dsDM["lat"].isel(trajectory=k).values
            mm      = ~(np.isnan(dm_x_km) | np.isnan(dm_y_km))
            axm.plot(
                dm_x_km[mm] * 1000,
                dm_y_km[mm] * 1000,
                color=c,
                alpha=0.5,
                linestyle="-",
                linewidth=0.8,
                transform=proj_ps,
                label=("DM  " + legend_labels[g] if  m_id == member_ids[0] else None),)

            # S2S member track
            s2s_x_km = dsS2S["lon"].isel(trajectory=k).values
            s2s_y_km = dsS2S["lat"].isel(trajectory=k).values
            mm       = ~(np.isnan(s2s_x_km) | np.isnan(s2s_y_km))
            axm.plot(
                s2s_x_km[mm] * 1000,
                s2s_y_km[mm] * 1000,
                color=c,
                alpha=0.5,
                linestyle="--",
                linewidth=0.8,
                transform=proj_ps,
                label=("S2S " + legend_labels[g] if m_id == member_ids[0] else None),)

        # Mark the release point once per group using the first member
        k0 = g * Nens
        dm_x0 = dsDM["lon"].isel(trajectory=k0).values
        dm_y0 = dsDM["lat"].isel(trajectory=k0).values
        mm0 = ~(np.isnan(dm_x0) | np.isnan(dm_y0))
        if np.any(mm0):
            axm.plot(dm_x0[mm0][0] * 1000, dm_y0[mm0][0] * 1000,
                     marker="o", color="k", markersize=10, transform=proj_ps)
            axm.plot(dm_x0[mm0][0] * 1000, dm_y0[mm0][0] * 1000,
                     marker="o", color=c, markersize=6, transform=proj_ps)

        # -------------------------
        # Time series: mean + 5th/95th percentile shading (km)
        # -------------------------
        t_dm, mean_dm, p05_dm, p95_dm = ensemble_distance_stats(dsDM, traj_slice, buoy_xy_km_df)
        axt.plot(t_dm, mean_dm, color=c, linestyle="-", linewidth=1)
        axt.fill_between(t_dm, p05_dm, p95_dm, color=c, alpha=0.2)

        t_s2s, mean_s2s, p05_s2s, p95_s2s = ensemble_distance_stats(dsS2S, traj_slice, buoy_xy_km_df)
        axt.plot(t_s2s, mean_s2s, color=c, linestyle="--", linewidth=1)
        axt.fill_between(t_s2s, p05_s2s, p95_s2s, color=c, alpha=0.2)

    # Figure parameters
    axt.set_ylim(0, 14)
    axt.set_xlim(t_ref_start, t_ref_end)

    midday_times = pd.date_range(t_ref_start.normalize(), t_ref_end.normalize(), freq="D") + pd.Timedelta(hours=12)
    for tmid in midday_times:
        axt.axvline(tmid, color="k", linewidth=2.0, alpha=0.35, zorder=0)

# Legend only once (top-left map)
ax_map["P108"].legend(loc="upper left", fontsize=9)
ax_map["P106"].legend(loc="upper left", fontsize=9)

ax_ts["P108"].tick_params(axis="x", which="both", labelbottom=False)
ax_ts["P108"].set_xlabel("")

# Panel labels (top-right): (a), (b), (c), (d)
panel_axes = [ax_map["P108"], ax_ts["P108"], ax_map["P106"], ax_ts["P106"]]
for i, ax in enumerate(panel_axes):
    ax.text(
        0.97,
        0.97,
        f"({chr(ord('a') + i)})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        zorder=10,
        bbox=dict(
            boxstyle="square,pad=0.2",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
        ),
    )

plt.savefig("/home/waynedj/Projects/swath_based_framework/figures/publication/Figure06_v001.png",dpi=500,bbox_inches="tight")
plt.close()

# %%
