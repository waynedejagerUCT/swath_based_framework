
#%%
"""
Unified SIC advection/decomposition for:
  - JAXA swath SIC (HDF5)
  - ECICE swath SIC (NetCDF) for TI, YI, FYI, and MYI

Outputs a single NetCDF with variables:
  obs_dSIC_<product>, adv_dSIC_<product>, resid_<product>
on the drift grid with lat/lon coordinates.

This script is intended to be run once (no pipeline requirements).
"""

import os
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
import xesmf as xe
from scipy.interpolate import griddata
from matplotlib.path import Path

from metpy.calc import divergence
from metpy.units import units


# -----------------------------------------------------------------------------
# Helpers (kept essentially as in your scripts)
# -----------------------------------------------------------------------------
def edges_from_centers_1d(c):
    c = np.asarray(c)
    dc = np.diff(c)
    if dc.size == 0:
        raise ValueError("Need at least 2 centers to compute edges.")
    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * dc[0]
    edges[-1] = c[-1] + 0.5 * dc[-1]
    return edges

def ps_xy_to_lonlat(x2d, y2d, epsg=3031):
    crs_ps = CRS.from_epsg(epsg)
    crs_ll = CRS.from_epsg(4326)
    tfm = Transformer.from_crs(crs_ps, crs_ll, always_xy=True)
    lon, lat = tfm.transform(x2d, y2d)
    lon = ((lon + 180) % 360) - 180
    return lon, lat

def lonlat_to_ps_xy(lon, lat, epsg=3031):
    crs_ll = CRS.from_epsg(4326)
    crs_ps = CRS.from_epsg(epsg)
    tfm = Transformer.from_crs(crs_ll, crs_ps, always_xy=True)
    x, y = tfm.transform(lon, lat)
    return x, y

def build_xesmf_grid_from_xy_centers(x1d_m, y1d_m, epsg=3031):
    x1d_m = np.asarray(x1d_m)
    y1d_m = np.asarray(y1d_m)

    xc2d, yc2d = np.meshgrid(x1d_m, y1d_m)
    lon, lat = ps_xy_to_lonlat(xc2d, yc2d, epsg=epsg)

    x_edges = edges_from_centers_1d(x1d_m)
    y_edges = edges_from_centers_1d(y1d_m)
    xb2d, yb2d = np.meshgrid(x_edges, y_edges)
    lon_b, lat_b = ps_xy_to_lonlat(xb2d, yb2d, epsg=epsg)

    grid = {"lon": lon, "lat": lat, "lon_b": lon_b, "lat_b": lat_b}
    return grid, xc2d, yc2d, xb2d, yb2d

def swath_to_grid_linear_interp(swath_lat, swath_lon, swath_sic_pct,
                                x1d_target_m, y1d_target_m,
                                epsg=3031, antarctic_only=True):
    """
    Swath -> target grid using linear interpolation of swath points in EPSG:3031.
    Returns SIC (%) on (y, x) target grid.
    """
    lat = np.asarray(swath_lat, dtype=float)
    lon = np.asarray(swath_lon, dtype=float)
    sic = np.asarray(swath_sic_pct, dtype=float)

    lon = ((lon + 180) % 360) - 180

    if antarctic_only:
        m = lat < 0.0
        lat = np.where(m, lat, np.nan)
        lon = np.where(m, lon, np.nan)
        sic = np.where(m, sic, np.nan)

    x_sw, y_sw = lonlat_to_ps_xy(lon, lat, epsg=epsg)

    valid = np.isfinite(x_sw) & np.isfinite(y_sw) & np.isfinite(sic)
    if np.count_nonzero(valid) < 10:
        ny = len(y1d_target_m)
        nx = len(x1d_target_m)
        return np.full((ny, nx), np.nan, dtype=float)

    points = np.column_stack((x_sw[valid].ravel(), y_sw[valid].ravel()))
    values = sic[valid].ravel()

    Xg, Yg = np.meshgrid(np.asarray(x1d_target_m), np.asarray(y1d_target_m))
    out = griddata(points, values, (Xg, Yg), method="linear")

    out = np.clip(out, 0.0, 100.0)
    return out

def swath_footprint_mask(swath_lat, swath_lon, x1d_target_m, y1d_target_m, epsg=3031):
    """
    Mask target grid cells that fall inside the swath perimeter (approximate).
    """
    lat = np.asarray(swath_lat, dtype=float)
    lon = np.asarray(swath_lon, dtype=float)
    lon = ((lon + 180.0) % 360.0) - 180.0

    x_sw, y_sw = lonlat_to_ps_xy(lon, lat, epsg=epsg)

    # perimeter polygon from swath edges
    top = np.column_stack((x_sw[0, :], y_sw[0, :]))
    right = np.column_stack((x_sw[1:, -1], y_sw[1:, -1]))
    bottom = np.column_stack((x_sw[-1, -2::-1], y_sw[-1, -2::-1]))
    left = np.column_stack((x_sw[-2:0:-1, 0], y_sw[-2:0:-1, 0]))
    poly = np.vstack((top, right, bottom, left))
    poly = poly[np.isfinite(poly).all(axis=1)]

    Xg, Yg = np.meshgrid(np.asarray(x1d_target_m), np.asarray(y1d_target_m))
    if poly.shape[0] < 3:
        return np.zeros_like(Xg, dtype=bool)

    path = Path(poly)
    pts = np.column_stack((Xg.ravel(), Yg.ravel()))
    return path.contains_points(pts).reshape(Xg.shape)

def conservative_regrid_with_coverage(src_field, src_grid, dst_grid, min_cov=0.9):
    """
    Conservative remap + coverage threshold masking.
    Keeps your “min_cov” behaviour.
    """
    src_da = xr.DataArray(src_field, dims=("y", "x"))
    cov_da = xr.DataArray(np.isfinite(src_field).astype(float), dims=("y", "x"))

    regridder = xe.Regridder(
        src_grid, dst_grid,
        method="conservative",
        periodic=False,
        reuse_weights=False
    )
    dst_field = regridder(src_da).data
    dst_cov = regridder(cov_da).data
    dst_field = np.where(dst_cov >= min_cov, dst_field, np.nan)
    return dst_field

# -----------------------------------------------------------------------------
# Product-specific readers (minimal branching)
# -----------------------------------------------------------------------------
PRODUCTS = {
    "jaxa": {
        "engine": "h5netcdf",
        "open_kwargs": {"phony_dims": "sort"},
        "lat_var": "Latitude of Observation Point",
        "lon_var": "Longitude of Observation Point",
        "sic_vars": {"SIC": "Geophysical Data"},  # single stream
        "raw_valid_min": 0,
        "raw_valid_max": 1000,  # tenths of %
        "raw_to_pct": lambda raw: raw / 10.0,    # 0..100 %
    },
    "ecice": {
        "engine": "netcdf4",
        "open_kwargs": {},
        "lat_var": "latitude",
        "lon_var": "longitude",
        "sic_vars": {
            "TI": "TI",
            "YI": "YI",
            "FYI": "FYI",
            "MYI": "MYI",
        },
        "raw_valid_min": 0,
        "raw_valid_max": 100,   # assumed already 0..100 %
        "raw_to_pct": lambda raw: raw * 1.0,     # already %
    }
}

def read_swath_sic_fraction(ds, product_key, stream_key, x1d_fine, y1d_fine, epsg=3031):
    """
    Returns SIC on fine grid as 0..1 fraction.
    """
    spec = PRODUCTS[product_key]
    sic_var = spec["sic_vars"][stream_key]
    lat_var = spec["lat_var"]
    lon_var = spec["lon_var"]

    sic_raw = ds[sic_var].data.squeeze()
    lat_sw  = ds[lat_var].data
    lon_sw  = ds[lon_var].data

    # range filter on raw
    mn, mx = spec["raw_valid_min"], spec["raw_valid_max"]
    sic_raw = np.where((sic_raw >= mn) & (sic_raw <= mx), sic_raw, np.nan)

    sic_pct_swath = spec["raw_to_pct"](sic_raw)  # 0..100 (%)

    sic_fine_pct = swath_to_grid_linear_interp(
        swath_lat=lat_sw,
        swath_lon=lon_sw,
        swath_sic_pct=sic_pct_swath,
        x1d_target_m=x1d_fine,
        y1d_target_m=y1d_fine,
        epsg=epsg,
    )

    # mask outside swath footprint to avoid “zeros/rings”
    sw_mask = swath_footprint_mask(lat_sw, lon_sw, x1d_fine, y1d_fine, epsg=epsg)
    sic_fine_pct = np.where(sw_mask, sic_fine_pct, np.nan)

    return (sic_fine_pct / 100.0).astype(float)

# -----------------------------------------------------------------------------
# USER PATHS (edit as needed)
# -----------------------------------------------------------------------------
FDIR_DRIFT  = "/home/waynedj/Data/seaicedrift/osisaf/swath/2019/07/27/"
f_drift     = os.path.join(FDIR_DRIFT,"icedrift_amsr2-gw1_tb37v-LapAtb37h-Lap_simplex_lev2_sh625_20190726165739w_20190727025045w.nc")
grid_res_km = 12.5
f_grid      = f"/home/waynedj/Data/grids/NSIDC0771_LatLon_PS_S{grid_res_km}km_v1.1.nc"
# JAXA inputs
FDIR_JAXA   = "/home/waynedj/Data/sic/jaxa_L2SIC/CaseStudy2019/"
f_jaxa_t0   = os.path.join(FDIR_JAXA, "GW1AM2_201907261709_107A_L2SGSICLC3300300.h5")
f_jaxa_t1   = os.path.join(FDIR_JAXA, "GW1AM2_201907270213_203D_L2SGSICLC3300300.h5")
# ECICE inputs
FDIR_ECICE  = "/home/waynedj/Data/intermediate/ecice/CaseStudy2019/L1R/res10/netcdf"
f_ecice_t0  = os.path.join(FDIR_ECICE, "GW1AM2_201907261709_107A_L1SGRTBR_2220220-ECICE.nc")
f_ecice_t1  = os.path.join(FDIR_ECICE, "GW1AM2_201907270213_203D_L1SGRTBR_2220220-ECICE.nc")
# Output
OUT_NC      = "/home/waynedj/Projects/swath_based_framework/Data/sic_decomposition_jaxa_ecice_TI_YI_FYI_MYI_on_driftgrid.nc"
# -----------------------------------------------------------------------------
# Load common datasets (drift + NSIDC grid)
# -----------------------------------------------------------------------------
ds_drift = xr.open_dataset(f_drift)
ds_grid  = xr.open_dataset(f_grid)

# Fixed fine grid (NSIDC)
x1d_fine = ds_grid["x"].data
y1d_fine = ds_grid["y"].data
fixed_grid, fine_xc2d, fine_yc2d, fine_xb2d, fine_yb2d = build_xesmf_grid_from_xy_centers(
    x1d_fine, y1d_fine, epsg=3031
)

# Drift grid
t0 = ds_drift["t0"].data[0]
t1 = ds_drift["t1"].data[0]
dt = (t1 - t0) / np.timedelta64(1, "s")  # seconds

drift_x_km = ds_drift["xc"].data
drift_y_km = ds_drift["yc"].data
drift_x_m  = drift_x_km * 1000.0
drift_y_m  = drift_y_km * 1000.0

drift_grid, drift_xc2d, drift_yc2d, drift_xb2d, drift_yb2d = build_xesmf_grid_from_xy_centers(
    drift_x_m, drift_y_m, epsg=3031
)
drift_lon2d = drift_grid["lon"]
drift_lat2d = drift_grid["lat"]

# Drift vectors
dX_km = ds_drift["driftX"].data[0]
dY_km = ds_drift["driftY"].data[0]

valid_drift = np.isfinite(dX_km) & np.isfinite(dY_km) & np.isfinite(dt) & (dt > 0)

U = np.where(valid_drift, (dX_km * 1000.0) / dt, np.nan)
V = np.where(valid_drift, (dY_km * 1000.0) / dt, np.nan)

# fill missing for displacement math; keep validity via mask later
U = np.where(np.isfinite(U), U, 0.0)
V = np.where(np.isfinite(V), V, 0.0)

dt_drift = np.array(dt, dtype=float)
dt_drift = np.where(np.isfinite(dt_drift) & (dt_drift > 0), dt_drift, np.nan)

# Divergence on drift grid (MetPy)
dx_m = float(np.nanmedian(np.diff(drift_x_m)))
dy_m = float(np.nanmedian(np.diff(drift_y_m)))

u_q = U * units("m/s")
v_q = V * units("m/s")

div_1s = divergence(u_q, v_q, dx=dx_m * units.meter, dy=dy_m * units.meter).to("1/s").magnitude
div_1s = np.where(valid_drift & np.isfinite(div_1s), div_1s, np.nan)


# -----------------------------------------------------------------------------
# Core processing for one product stream
# -----------------------------------------------------------------------------
def process_stream(ds0, ds1, product_key, stream_key, min_cov=0.9):
    """
    Returns obs_dSIC, adv_dSIC, resid on drift grid (percent units).
    """
    # Swath -> fine (0..1)
    sic0_fine = read_swath_sic_fraction(ds0, product_key, stream_key, x1d_fine, y1d_fine, epsg=3031)
    sic1_fine = read_swath_sic_fraction(ds1, product_key, stream_key, x1d_fine, y1d_fine, epsg=3031)

    # Fine -> drift (conservative)
    sic0_drift = conservative_regrid_with_coverage(sic0_fine, fixed_grid, drift_grid, min_cov=min_cov)
    sic1_drift = conservative_regrid_with_coverage(sic1_fine, fixed_grid, drift_grid, min_cov=min_cov)

    sic0_drift = np.clip(sic0_drift, 0.0, 1.0)
    sic1_drift = np.clip(sic1_drift, 0.0, 1.0)

    # Build upstream grid (pull advection with rigid translation)
    dx = U * dt_drift
    dy = V * dt_drift

    # centers
    xc_up = drift_xc2d - dx
    yc_up = drift_yc2d - dy
    lon_up, lat_up = ps_xy_to_lonlat(xc_up, yc_up, epsg=3031)

    # corners (padded)
    dx_b = np.pad(dx, ((0, 1), (0, 1)), mode="edge")
    dy_b = np.pad(dy, ((0, 1), (0, 1)), mode="edge")
    xb_up = drift_xb2d - dx_b
    yb_up = drift_yb2d - dy_b
    lon_b_up, lat_b_up = ps_xy_to_lonlat(xb_up, yb_up, epsg=3031)

    upstream_grid = {"lon": lon_up, "lat": lat_up, "lon_b": lon_b_up, "lat_b": lat_b_up}

    # Advect (conservative remap)
    sic01_drift = conservative_regrid_with_coverage(sic0_drift, drift_grid, upstream_grid, min_cov=min_cov)
    sic01_drift = np.clip(sic01_drift, 0.0, 1.0)

    # Divergence correction: C = C_adv * exp(-div * dt)
    sic01_divcorr = sic01_drift * np.exp(-div_1s * dt_drift)
    sic01_divcorr = np.where(np.isfinite(sic01_divcorr), sic01_divcorr, np.nan)
    sic01_divcorr = np.clip(sic01_divcorr, 0.0, 1.0)

    # Masks
    # obs_dSIC: requires only valid SIC at t0 and t1 (swath coverage + remap coverage)
    obs_mask = np.isfinite(sic0_drift) & np.isfinite(sic1_drift)

    # adv/resid: additionally require valid drift and valid advected field
    t01_mask = obs_mask & np.isfinite(sic01_divcorr) & valid_drift

    # Diagnostics (%)
    obs_dSIC = np.where(obs_mask, (sic1_drift - sic0_drift) * 100.0, np.nan)
    adv_dSIC = np.where(t01_mask, (sic01_divcorr - sic0_drift) * 100.0, np.nan)
    resid    = np.where(t01_mask, (sic1_drift - sic01_divcorr) * 100.0, np.nan)

    # IMPORTANT:
    # Do NOT apply valid_drift to obs_dSIC (it’s observational / swath-driven).
    # Keep drift masking for adv_dSIC and resid (already enforced by t01_mask).
    return obs_dSIC.astype(np.float32), adv_dSIC.astype(np.float32), resid.astype(np.float32)


# -----------------------------------------------------------------------------
# Load swath datasets and run all streams
# -----------------------------------------------------------------------------
# JAXA
spec_j = PRODUCTS["jaxa"]
ds_j0 = xr.open_dataset(f_jaxa_t0, engine=spec_j["engine"], **spec_j["open_kwargs"])
ds_j1 = xr.open_dataset(f_jaxa_t1, engine=spec_j["engine"], **spec_j["open_kwargs"])

# ECICE
spec_e = PRODUCTS["ecice"]
ds_e0 = xr.open_dataset(f_ecice_t0, engine=spec_e["engine"], **spec_e["open_kwargs"])
ds_e1 = xr.open_dataset(f_ecice_t1, engine=spec_e["engine"], **spec_e["open_kwargs"])

# Compute outputs
min_cov = 0.9

obs_jaxa, adv_jaxa, resid_jaxa = process_stream(ds_j0, ds_j1, "jaxa", "SIC", min_cov=min_cov)
obs_ti,   adv_ti,   resid_ti   = process_stream(ds_e0, ds_e1, "ecice", "TI",  min_cov=min_cov)
obs_yi,   adv_yi,   resid_yi   = process_stream(ds_e0, ds_e1, "ecice", "YI",  min_cov=min_cov)
obs_fyi,  adv_fyi,  resid_fyi  = process_stream(ds_e0, ds_e1, "ecice", "FYI", min_cov=min_cov)
obs_myi,  adv_myi,  resid_myi  = process_stream(ds_e0, ds_e1, "ecice", "MYI", min_cov=min_cov)


# -----------------------------------------------------------------------------
# Save to NetCDF (drift grid, with lat/lon)
# -----------------------------------------------------------------------------
out = xr.Dataset(
    data_vars={
        "obs_dSIC_jaxa": (("y", "x"), obs_jaxa),
        "adv_dSIC_jaxa": (("y", "x"), adv_jaxa),
        "resid_jaxa":    (("y", "x"), resid_jaxa),

        "obs_dSIC_ecice_TI": (("y", "x"), obs_ti),
        "adv_dSIC_ecice_TI": (("y", "x"), adv_ti),
        "resid_ecice_TI":    (("y", "x"), resid_ti),

        "obs_dSIC_ecice_YI": (("y", "x"), obs_yi),
        "adv_dSIC_ecice_YI": (("y", "x"), adv_yi),
        "resid_ecice_YI":    (("y", "x"), resid_yi),

        "obs_dSIC_ecice_FYI": (("y", "x"), obs_fyi),
        "adv_dSIC_ecice_FYI": (("y", "x"), adv_fyi),
        "resid_ecice_FYI":    (("y", "x"), resid_fyi),

        "obs_dSIC_ecice_MYI": (("y", "x"), obs_myi),
        "adv_dSIC_ecice_MYI": (("y", "x"), adv_myi),
        "resid_ecice_MYI":    (("y", "x"), resid_myi),
    },
    coords={
        # keep your drift-grid indexing and provide 2D lat/lon
        "x": (("x",), drift_x_m.astype(np.float64)),
        "y": (("y",), drift_y_m.astype(np.float64)),
        "lon": (("y", "x"), drift_lon2d.astype(np.float64)),
        "lat": (("y", "x"), drift_lat2d.astype(np.float64)),
    },
    attrs={
        "description": "SIC decomposition on OSI-SAF drift grid: observed change, advected+divergence change, residual",
        "projection": "EPSG:3031 (South Polar Stereographic) for advection; lon/lat provided for plotting",
        "min_cov": float(min_cov),
    }
)

# Add units
for v in list(out.data_vars):
    out[v].attrs["units"] = "percent"
out["lon"].attrs["units"] = "degrees_east"
out["lat"].attrs["units"] = "degrees_north"
out["x"].attrs["units"]   = "m"
out["y"].attrs["units"]   = "m"

# If file exists, delete it before writing new one
if os.path.exists(OUT_NC):
    os.remove(OUT_NC)
    print(f"Existing file removed: {OUT_NC}")

out.to_netcdf(OUT_NC)
print(f"Saved: {OUT_NC}")


#%%
