# Swath-Based Sea-Ice Advection Framework

## Overview

This repository contains Python scripts and supporting datasets used in the manuscript:

**“Dynamic Decomposition of Sub-Daily Antarctic Sea-Ice Concentration Variability using Passive Microwave Swath Observations.”**

The framework performs two primary tasks:

1. **Sea-ice concentration retrieval** using the ECICE Ice Type algorithm applied to AMSR2 brightness temperature observations (19H, 19V, 37H, 37V).
2. **Advection of swath-derived sea-ice concentration fields** using swath-to-swath sea-ice motion products.

The workflow integrates satellite brightness temperature observations, sea-ice motion datasets, and derived sea-ice concentration fields to analyze Antarctic sea-ice evolution at sub-daily timescales.

---

# Repository Structure

## Advection Scripts

* `AdvectSIC_CourseScale_v008.py`
  Main script for advecting sea-ice concentration fields at **62.5 km resolution**.

## OceanParcels Execution

* `execute_OceanParcels_v003.py`
  Executes the particle advection framework using the **OceanParcels** library.

## Data Preparation

* `prepare_ecice_input_v002.py`
  Converts AMSR2 Level-1R brightness temperature data into the input format required for ECICE processing.

* `prepare_ecice_output_v002.py`
  Converts ECICE output tables into georeferenced **NetCDF** datasets for analysis.

---

# Generated Data

## Parcel Ensemble Trajectories

Combined drift datasets are created using the following file lists:

```
WeddellCaseStudy2019_24HRDRIFT.txt
WeddellCaseStudy2019_S2SDRIFT.txt
```

Running parcel simulations for **P106** and **P108** using these drift datasets produces:

```
TrajectoryData_CaseStudy2019_Daily_P106_v004.zarr
TrajectoryData_CaseStudy2019_Daily_P108_v004.zarr
TrajectoryData_CaseStudy2019_Swath_P106_v004.zarr
TrajectoryData_CaseStudy2019_Swath_P108_v004.zarr
```

The script `AdvectSIC_CourseScale_v008.py` introduces perturbations to the drift vector field for each particle in order to generate an **ensemble spread**. Perturbation magnitudes are defined by the motion product uncertainties described in the manuscript.

---

# Workflow

## 1. ECICE Retrieval

1. Download **AMSR2 Level-1R brightness temperature datasets**.

2. Prepare ECICE input files:

```
python prepare_ecice_input_v002.py
```

This script accepts `.h5` files and outputs `.txt` tables where **each row represents one pixel**.

3. Run the ECICE retrieval:

```
python execute_ecice_v001.py
```

4. Convert ECICE output to NetCDF:

```
python prepare_ecice_output_v002.py
```

This converts `.txt` output tables into `.nc` files in the requested projection.

---

## 2. Lagrangian Backward / Pulling Transform

1. Load sea-ice concentration datasets at **t₀** and **t₁**.
2. Interpolate the fields onto the swath-to-swath grid using **linear interpolation**.
3. Advect the SIC field at **t₀** using the corresponding swath-to-swath motion dataset:

```
python AdvectSIC_CourseScale_v008.py
```

---

# Data Sources

The datasets required to run this framework are publicly available from the following sources.

### Satellite Brightness Temperatures

Level-1R **AMSR2 brightness temperatures** from the **GCOM-W1 satellite** are available through the JAXA G-Portal:

https://gportal.jaxa.jp

---

### Antarctic Polynya Dataset

Probability distributions used in this study were derived from sample sites described in **Melsheimer et al. (2023)**, which utilize the Antarctic polynya dataset from:

https://www.cen.uni-hamburg.de/en/icdc/data/cryosphere/polynya-antarctic.html

(Kern et al., 2007; ICDC/CEN, University of Hamburg)

---

### Sea-Ice Motion and Concentration Products

Provided by the **EUMETSAT Ocean and Sea Ice Satellite Application Facility (OSI SAF)** and distributed via the **Arctic Data Centre**.

* Global Sea Ice Concentration (AMSR2) — OSI-408-a
* Global Low Resolution Sea Ice Drift Data Record — OSI-455

---

### Sea-Ice Concentration Products

Daily **ECICE** and **ASI** sea-ice concentration datasets are provided by the **Institute of Environmental Physics, University of Bremen**:

https://data.seaice.uni-bremen.de/

---

### Surface Velocity Profiler Data

Surface velocity profiler observations (**P106 and P108**) were deployed by the **Alfred Wegener Institute**.

---

# Dataset Citations

If using this framework or associated datasets, please cite:

* EUMETSAT OSI SAF (2016). *Global Sea Ice Concentration (AMSR2), OSI-408-a.*
  https://doi.org/10.15770/EUM_SAF_OSI_NRT_2023

* EUMETSAT OSI SAF (2022). *Global Low Resolution Sea Ice Drift Data Record 1991–2020 (Version 1), OSI-455.*
  https://doi.org/10.15770/EUM_SAF_OSI_0012

* Kern, S., Spreen, G., Kaleschke, L., de la Rosa, S., & Heygster, G. (2007).
  Polynya signature simulation method. *Annals of Glaciology.*
  https://doi.org/10.3189/172756407782871585

* Lavergne, T. (2020). *Antarctic sea-ice drift vectors using a swath-to-swath approach.*
  Norwegian Meteorological Institute.
  https://doi.org/10.21343/0asd-6t60

* Melsheimer, C., & Spreen, G. (2019). *AMSR2 ASI sea ice concentration data, Antarctic.*
  https://doi.org/10.1594/PANGAEA.898400

* Melsheimer, C., & Spreen, G. (2022). *Uncorrected Ice Type Concentrations (ECICE).*
  University of Bremen.

* Nicolaus, M. et al. (2017). *Sea ice drift and meteorological observations from surface velocity profilers.*
  https://doi.org/10.1594/PANGAEA.875652

---

# Notes

* Large satellite datasets are **not included** in this repository.
* Users must download required datasets independently.
* File paths may need to be modified depending on local directory structure.

---

# Author

Wayne de Jager
Department of Oceanography
University of Cape Town

---

# License

This repository is intended for **research use**.
