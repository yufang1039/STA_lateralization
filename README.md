# STA_lateralization

Code supporting analysis of MEG source time series at subcortical STA coordinates, Shen-atlas label time courses, and left vs right hemispheric comparisons.

## File descriptions

1. `Figure 5/`: folder containing scripts and demo data for Figure 5 analysis (STA source estimation and surface plotting)
2. `Figure 5/script/MNI_source_estimation_freq.py`: Python script to compute LCMV source time series at STA MNI coordinates
3. `Figure 5/script/shen_atlas_source_estimation_freq.py`: Python script to extract Shen-268 atlas label time courses using LCMV beamforming
4. `Figure 5/script/get_envelope_sta_shen.py`: Python script to compute orthogonalized log-envelope correlations between STA and Shen label time series
5. `Figure 5/script/Figure_5C_plot_shen_surface.py`: Python script to generate surface visualizations of significant STA–brain correlations
6. `comparison_left_right.py`: Python script to perform group-level left–right statistical comparisons and cluster-based permutation tests

## Installation

Download the repository and you're good to go. Read the comments and documentation within each code for usage guidance.

## Data Setup

Before running the scripts, you need to download the required subject data:

1. Download a single subject's directory from: https://osf.io/search
2. Place the downloaded subject directory in the `Figure 5/data/` folder
3. The subject directory should contain the necessary MRI files and MEG data in the expected structure (see `demo_MNI_source_estimation_freq.py` for the expected file organization)

## Dependencies

1. Install the following Python libraries (e.g., via pip): mne, mne_connectivity, numpy, scipy, pandas, matplotlib, seaborn, nibabel, nilearn, tqdm, joblib
2. Ensure FreeSurfer and MNE-Python environments are properly configured if running source estimation from scratch.

## Compatibility

The codes have been tested on standard Linux environments with Python 3.x.
