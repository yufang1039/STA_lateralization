## STA_lateralization

Code for estimating MEG source time series at STA coordinates, extracting
Shen-atlas label time courses, computing STA–to–brain envelope correlations, and
comparing left vs right STA's connection with whole brain.


### File descriptions

1. `MNI_source_estimation_freq.py`  
   Compute LCMV source time series at predefined left/right STA MNI coordinates
   (per subject, per frequency) and output
   `right_STA_ts_lcmv_{freq}Hz.npy` / `left_STA_ts_lcmv_{freq}Hz.npy`.

2. `shen_atlas_source_estimation_freq.py`  
   Compute Shen-268 atlas label time courses using LCMV beamforming in a volume
   source space, outputting `shen_label_tc_lcmv_{freq}Hz.npy` for each subject.

3. `get_envelope_sta_shen.py`  
   Compute orthogonalized log-envelope correlations between STA time series
   (left/right, 5/20 Hz) and Shen label time courses for each subject, outputting
   `{side}_sta_label_envelope_corr_{freq}Hz.npy`.

4. `comparison_left_right.py`  
   Load STA–label envelope correlations across subjects, perform left–right
   comparisons (summary statistics, region-by-region tests, cluster-based
   permutation analysis), and generate group-level figures and CSV summaries.

### Typical pipeline

1. Run `MNI_source_estimation_freq.py` to generate STA time series per subject
   and frequency.
2. Run `shen_atlas_source_estimation_freq.py` to generate Shen label time
   courses.
3. Run `get_envelope_sta_shen.py` to compute STA–label envelope correlations
   (left/right, 5/20 Hz).
4. Run `comparison_left_right.py` for group-level left–right analyses and
   visualizations.

### Installation / dependencies

Download or clone this project and ensure the following Python libraries are
available (e.g., via `pip` or your HPC module system):

- `mne`, `mne_connectivity`
- `numpy`, `scipy`, `pandas`
- `matplotlib`, `seaborn`
- `nibabel`, `nilearn`
- `tqdm`, `joblib`

Several scripts use `multiprocessing.Pool` and set `OMP_NUM_THREADS`; on an HPC
system, match process and thread counts to your allocated resources.



