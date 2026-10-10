# README.md

This project implements a statistical analysis pipeline based on `pyhf` to estimate upper limits inspired by the LHCb Run 2 analysis for the tau23mu decay.

---

## File Overview

### 1. `parameters.py`
* **What it does**: Gathers all global constants, physical parameters, and analysis configurations.
* **Key contents**: 
  * Mass constants and experiment sensitivity (e.g., tau mass, resolutions, category ranges).
  * Definition of the Signal Region (`SR_MIN`, `SR_MAX`) and the total mass window (`MASS_MIN`, `MASS_MAX`).
  * Toy data settings (e.g., `SEED`, `POI` for the signal).

### 2. `toy_generator.py`
* **What it does**: Generates event-level unbinned toy datasets for each classifier category.
* **Key contents**: 
  * Statistical sampling of the combinatorial background (exponential distribution).
  * Sampling of the signal (Gaussian distribution) 
  * Sampling of the reflection components.
  * Optional visualization functions to plot generated toy histograms globally and per category.

### 3. `utils.py`
* **What it does**: Provides mathematical support functions, integration tools, and data manipulation routines.
* **Key contents**: 
  * Analytical calculation of integrals for exponential and Gaussian functions (`exp_integral`, `gauss_integral`).
  * Binning management for raw data into category-specific histograms (`binning`).
  * Support functions to compute binned expected counts for background, signal, and reflections.

### 4. `stats.py`
* **What it does**: Manages statistical tools within the framework.
* **Key contents**: 
  * Implementation of the `iminuit` fit (`fit_background_sidebands`) based on unbinned maximum likelihood on the sidebands, extracting parameters via **Migrad**, standard errors via **Hesse**, and asymmetric errors via **Minos**.
  * The `upper_limit` function for calculating upper limits using `pyhf`, supporting Brazil-style plots.

### 5. `models.py`
* **What it does**: Builds expected statistical models and workspaces compatible with `pyhf`.
* **Key contents**: 
  * Manages background models (`flat`, `exp`, `refl`).
  * Assembles channels, observations, and statistical modifiers (`normsys`, `histosys`, `normfactor`) to generate the final JSON workspace required by `pyhf`.

### 6. `analysis_SR.py`
* **What it does**: Main executable script focused on the Signal Region (SR).
* **Key contents**: 
  * Generates toy data, extracts expected models limited to the signal region, and calculates upper limits per category or combined (total).

### 7. `analysis_TOT.py`
* **What it does**: Main executable script extended to the full mass window.
* **Key contents**: 
  * Executes the same pipeline as `analysis_SR.py`, but considers the full interval (`MASS_MIN` to `MASS_MAX`) for binning and background sideband fitting[cite: 2].
