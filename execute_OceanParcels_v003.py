#%%
import os
import shutil
from parcels import Field, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4
import pandas as pd
from datetime import timedelta
import xarray as xr
import glob
import numpy as np
from pyproj import Transformer
from parcels import ParcelsRandom


def transform_latlon_to_xy(lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6932", always_xy=True)
    x_m, y_m = transformer.transform(lon, lat)
    x_km = np.array(x_m) / 1000.0
    y_km = np.array(y_m) / 1000.0
    return x_km, y_km

def AddDriftUncertainty(particle, fieldset, time):
    """Add Gaussian velocity perturbation using 1-sigma component uncertainty (km/s).
    Applied as an extra displacement over the timestep (dt)."""

    su = fieldset.U_unc[time, particle.depth, particle.lat, particle.lon]
    sv = fieldset.V_unc[time, particle.depth, particle.lat, particle.lon]

    du = ParcelsRandom.normalvariate(0.0, su)  # km/s
    dv = ParcelsRandom.normalvariate(0.0, sv)  # km/s

    # mesh="flat" => lon/lat treated as x/y in km
    particle.lon += du * particle.dt
    particle.lat += dv * particle.dt


#setup
DRIFT_DATASET  = "Daily" # 'Daily' or 'Swath'
DRIFT_DATA_v   = 'v001'
output_version = 'v004'
P_list         = ['P106', 'P108']
time_t0        = pd.Timestamp("2019-07-26 12:00:00")
time_t1        = pd.Timestamp("2019-07-30 12:00:00")
timestamps     = np.array(pd.date_range(start=time_t0, end=time_t1, freq="1D"))

# Swath uncertainty proxy from literature: RMSE ~1.5 km over 24 h -> 1.5/86400 km/s
SWATH_SIGMA_KMPS = 1.04e-5  # km/s

#fetch drift data
velocity_file = f'/home/waynedj/Projects/swath_based_framework/Data/Combined{DRIFT_DATASET}DriftDataset_v001.nc'
filenames     = {'U': velocity_file, 'V': velocity_file}
variables     = {'U':'U','V':'V'}
dimensions    = {'lat':'y','lon':'x', 'time':'time'}

if DRIFT_DATASET.lower() == "daily":
    filenames.update({'U_unc': velocity_file, 'V_unc': velocity_file})
    variables.update({'U_unc': 'uncert', 'V_unc': 'uncert'})

# setup FieldSet
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh="flat")

# if Swath, add constant uncertainty fields after FieldSet creation ---
if DRIFT_DATASET.lower() == "swath":
    fieldset.add_constant_field("U_unc", SWATH_SIGMA_KMPS, mesh="flat")
    fieldset.add_constant_field("V_unc", SWATH_SIGMA_KMPS, mesh="flat")

for P in P_list:
    print("-------------------------------")
    print(f"Processing {P} from {DRIFT_DATASET} dataset...")
    print("-------------------------------")

    #fetch buoy data and extract lat/lon for each timestamp
    P_fname     = glob.glob(f'/home/waynedj/Data/seaicedrift/AWIbuoys/2019{P}_*.csv')
    df_P        = pd.read_csv(P_fname[0], parse_dates=['time'])
    buoy_lat      = []
    buoy_lon      = []
    buoy_depth    = []
    for timestamp in timestamps:
        time_t0 = timestamp
        closest_idx = (df_P['time'] - time_t0).abs().idxmin()
        closest_row = df_P.loc[[closest_idx]]
        buoy_lat.append(closest_row['latitude (deg)'].iloc[0])
        buoy_lon.append(closest_row['longitude (deg)'].iloc[0])
        buoy_depth.append(0)

    #convert lat/lon to x/y in km for Parcels
    x_km, y_km = transform_latlon_to_xy(buoy_lat, buoy_lon)

    # ensemble members per release time
    Nens = 100
    lon_ens   = np.repeat(x_km, Nens)
    lat_ens   = np.repeat(y_km, Nens)
    time_ens  = np.repeat(timestamps, Nens)
    depth_ens = np.repeat(buoy_depth, Nens)

    # ParticleSet initialization
    pset = ParticleSet.from_list(
        fieldset = fieldset,
        pclass   = JITParticle,
        lon      = lon_ens,
        lat      = lat_ens,
        time     = time_ens,
        depth    = depth_ens)

    fname_zarr = f"/home/waynedj/Projects/swath_based_framework/Data/TrajectoryData_CaseStudy2019_{DRIFT_DATASET}_{P}_{output_version}.zarr"

    # Clean up old runs to prevent Zarr conflicts
    if os.path.exists(fname_zarr):
        shutil.rmtree(fname_zarr)

    # execute with combined advection and uncertainty kernels
    kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(AddDriftUncertainty)
    output_file = pset.ParticleFile(name=fname_zarr, outputdt=timedelta(minutes=60))
    pset.execute(
        kernels,
        runtime=timedelta(days=4),
        dt=timedelta(minutes=30),
        output_file=output_file)

    #Finalize output for different Parcels versions before reading
    if hasattr(output_file, "close"):
        output_file.close()
    elif hasattr(output_file, "export"):
        output_file.export()
    
        #Finalize output for different Parcels versions before reading
    if hasattr(output_file, "close"):
        output_file.close()
    elif hasattr(output_file, "export"):
        output_file.export()


# %%
