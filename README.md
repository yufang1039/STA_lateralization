## STA_lateralization – Pipeline Overview

This folder contains four main scripts that implement a complete pipeline for
computing MEG source time series at subcortical STA coordinates, extracting
Shen-atlas label time courses, computing STA–to–brain envelope correlations, and
performing left–right statistical comparisons and visualization.

The intended execution order is:

1. `MNI_source_estimation_freq.py`
2. `shen_atlas_source_estimation_freq.py`
3. `get_envelope_sta_shen.py`
4. `comparison_left_right.py`

All scripts assume the Cam-CAN / FreeSurfer directory layout under
`/projectnb/viscog01/yufang/camcan/cc700/freesurfer_output` and use subject IDs
from `full.txt` unless otherwise modified.

---

## 1. `MNI_source_estimation_freq.py`

**Purpose:**  
Compute LCMV beamformer source time series at two predefined STA MNI
coordinates (right and left) for each subject and frequency band, and save the
resulting source signals as `.npy` files.

**Key functionality:**

- `MNI_to_MRI(...)` converts MNI (mm) coordinates to FreeSurfer MRI
  (surface RAS, meters) using `read_ras_mni_t` and the subject’s `T1.mgz`.
- `compute_SourceSpace(...)` builds a *volume* source space (default spacing
  7.8 mm) based on the inner skull surface.
- `forward_model(...)` computes a single-layer BEM and MEG forward solution.
- `run_coordinate_time_series(...)`:
  - Computes data covariance and LCMV filters.
  - Applies Hilbert transform and `apply_lcmv_raw` to obtain a source estimate.
  - Finds the nearest grid points to the (right/left) STA MRI coordinates in
    the source space and extracts their time series.
  - Saves:
    - `right_STA_ts_lcmv_{freq}Hz.npy`
    - `left_STA_ts_lcmv_{freq}Hz.npy`
    under `<subjects_dir>/<subject>/mri/sta_time_series/`.
- `run_subject_in_parallel(...)`:
  - Loads raw MEG (`*_ses-rest_task-rest.fif`).
  - Creates / reuses volume source space and forward solution.
  - Computes ECG/EOG projectors if missing, applies them, and saves a projected
    raw file.
  - High‑pass filters at 0.5 Hz, computes covariance, and band‑pass filters
    around a requested center frequency (e.g. 5, 20 Hz).
  - Calls `run_coordinate_time_series(...)`.

**Main entry point:**

- The current `main()`:
  - Reads `case_list` from `full.txt` but overrides it with a fixed subset of
    subjects in `case_list`.
  - Uses a `cutoff_date` (2025‑10‑27) and `get_files_before_cutoff(...)` to
    decide which `(subject, freq)` pairs to (re)process based on existing
    `sta_time_series` files.
  - Spawns a multiprocessing pool (16 processes) and calls
    `run_subject_in_parallel(...)` for qualifying subjects and frequencies.

**Typical usage:**

```bash
cd /projectnb/viscog01/yufang/Neural_representaion_proj/STA_lateralization
python MNI_source_estimation_freq.py
```

Make sure that:

- `cases` and `subjects_dir` paths at the bottom of the script match your data.
- The Cam-CAN `*-trans.fif` coregistration files exist for each subject.

---

## 2. `shen_atlas_source_estimation_freq.py`

**Purpose:**  
Compute Shen-268 atlas label time courses from MEG using LCMV beamforming in a
volume source space and save a single `.npy` time‑course array per subject and
frequency.

**Key functionality:**

- `compute_SourceSpace(...)` and `forward_model(...)`:
  - Mirror the setup in `MNI_source_estimation_freq.py` to create / reuse a
    volume source space and forward model (BEM + MEG forward).
- `run_label_time_course(...)`:
  - Computes data covariance and LCMV filters.
  - Applies Hilbert transform and `apply_lcmv_raw` to obtain source estimates.
  - Uses a Shen atlas volume `shen_freesurfer.mgz` in each subject’s `mri`
    directory along with `label_map` (mapping `"1".. "268"` → `1..268`) to call
    `mne.extract_label_time_course(...)` with `mri_resolution=True`.
  - Saves `shen_label_tc_lcmv_{freq}Hz.npy` under
    `<subjects_dir>/<subject>/mri/shen_label_time_course/`.
- `run_subject_in_parallel(...)`:
  - Very similar preprocessing chain as in `MNI_source_estimation_freq.py`:
    ECG/EOG projectors, projection application, 0.5 Hz high‑pass filter, raw
    covariance, frequency‑specific band‑pass filter, then
    `run_label_time_course(...)`.

**Main entry point:**

- The script:
  - Reads `full_case_list` from `full.txt`, divides it into 6 equal portions,
    and selects portion `i` (currently `i = 1`) to form `case_list`.
  - Defines `label_map` for the 268 Shen regions.
  - `main()`:
    - Uses `volume_spacing = 7.8`.
    - For `freq` in `[10]` (currently fixed to 10 Hz):
      - For each subject in `case_list`, checks if
        `shen_label_tc_lcmv_{freq}Hz.npy` exists; if not, schedules
        `run_subject_in_parallel(...)` in a multiprocessing pool (8 processes).

**Typical usage:**

```bash
cd /projectnb/viscog01/yufang/Neural_representaion_proj/STA_lateralization
python shen_atlas_source_estimation_freq.py
```

Before running, verify:

- `cases` / `subjects_dir` paths.
- That `shen_freesurfer.mgz` exists for each subject under `mri/`.
- That any subject list partitioning (`i`, `total_portions`) matches your
  job‑array or batch strategy.

---

## 3. `get_envelope_sta_shen.py`

**Purpose:**  
For each subject, hemisphere (left/right STA), and frequency (5, 20 Hz), compute
orthogonalized log‑envelope correlations between the STA time series (from
`MNI_source_estimation_freq.py`) and Shen label time courses (from
`shen_atlas_source_estimation_freq.py`). Save per‑subject STA–label correlation
vectors.

**Key functionality:**

- `process_subject_freq_side(subjects_dir, subject, freq, side)`:
  - Expects:
    - STA time series:
      - `<subjects_dir>/<subject>/mri/sta_time_series/{side}_STA_ts_lcmv_{freq}Hz.npy`
    - Shen label time courses:
      - `<subjects_dir>/<subject>/mri/shen_label_time_course/shen_label_tc_lcmv_{freq}Hz.npy`
  - Creates an output folder:
    - `<subjects_dir>/<subject>/mri/shen_label_envelope_corr/`
  - If output already exists (`{side}_sta_label_envelope_corr_{freq}Hz.npy`),
    the subject/side/frequency is skipped.
  - Loads the STA and label time series, reshapes them to the format expected by
    `mne_connectivity.envelope_correlation` (channels × time; wrapped in a
    list).
  - Calls `envelope_correlation(..., orthogonalize="pairwise", log=True,
    absolute=True)` and extracts correlations between the STA signal (index 0)
    and each Shen label.
  - Saves a 1D array:
    - `{side}_sta_label_envelope_corr_{freq}Hz.npy`
    where length ≈ 268 (one value per Shen region).
- `get_subjects_with_incomplete_files(...)`:
  - Scans all subjects for the presence of up to 4 expected STA–label
    correlation files (left/right × 5/20 Hz) in `shen_label_envelope_corr`.
  - Returns a list of subjects with fewer than 4 files, and prints a summary.
- `main()`:
  - Sets `cases` and `subjects_dir` (both pointing to the Cam-CAN
    freesurfer_output tree).
  - Calls `get_subjects_with_incomplete_files(...)` to create an initial
    `case_list` (but then currently re‑reads `case_list` from `cases`, i.e.,
    processes all subjects; this can be adjusted).
  - Optionally allows list partitioning into 6 portions (`i` variable) – this
    is currently commented out.
  - Builds a list of tasks:
    - `(subjects_dir, subject, freq, side)` for
      `freq ∈ {5, 20}` and `side ∈ {"left", "right"}` where:
      - Output file does not yet exist, **and**
      - Both STA and label input files exist.
  - Uses a multiprocessing pool (up to 8 processes) to run
    `process_wrapper(...)` over all tasks.

**Typical usage:**

```bash
cd /projectnb/viscog01/yufang/Neural_representaion_proj/STA_lateralization
python get_envelope_sta_shen.py
```

Run this **after**:

1. `MNI_source_estimation_freq.py` has generated `*_STA_ts_lcmv_*Hz.npy`.
2. `shen_atlas_source_estimation_freq.py` has generated
   `shen_label_tc_lcmv_*Hz.npy`.

---

## 4. `comparison_left_right.py`

**Purpose:**  
Perform group‑level analysis of STA–brain envelope correlations, comparing left
vs right STA across subjects, frequencies (5 Hz, 20 Hz), and Shen regions.
Includes:

- Descriptive statistics and paired t‑tests for hemispheric averages.
- Region‑by‑region paired tests with multiple‑comparison corrections.
- Cluster‑based permutation tests using a Shen atlas NIfTI volume.
- Multiple visualization routines (boxplots, cluster maps, difference maps,
  STA–correlation maps).

**Inputs and configuration:**

- Subject list: `cases = .../freesurfer_output/full.txt`
- Data base directory: `subjects_dir = .../freesurfer_output`
- Results base directory: `results_dir` (default:
  `/projectnb/viscog01/yufang/Neural_representaion_proj/results`)
- Correlation files expected for each subject in:
  - `<base_dir>/<subject>/mri/shen_label_envelope_corr/`:
    - `left_sta_label_envelope_corr_5Hz.npy`
    - `right_sta_label_envelope_corr_5Hz.npy`
    - `left_sta_label_envelope_corr_20Hz.npy`
    - `right_sta_label_envelope_corr_20Hz.npy`
- Atlas volume for cluster tests and visualizations:
  - `shen_2mm_268_parcellation.nii.gz`

`SAVE_RESULTS` controls whether outputs are written to disk (CSV, PDFs, text
summary). `N_PERMUTATIONS` sets the number of permutations for cluster tests.
A fixed set of subjects can be excluded via `excluded_subjects`.

**Main analysis steps:**

1. **Data loading and subject filtering**
   - Reads `case_list`, removes excluded subjects.
   - For each subject, checks for all 4 required STA–label correlation files.
     Subjects with missing files are skipped.
   - Loads each STA–label correlation file, squeezes to 1D arrays, and extracts
     STA–to–brain correlations (one value per 268 Shen regions). These are
     stored as:
     - `left_5hz_regions`, `right_5hz_regions`,
     - `left_20hz_regions`, `right_20hz_regions` (arrays with shape
       `n_subjects × 268`).
   - Also computes and stores per‑subject summary measures (mean, std, median
     correlations).

2. **Analysis 1 – Average correlation across all regions**
   - Computes group mean, SD, and median for left vs right STA at 5 Hz and
     20 Hz.
   - Runs paired t‑tests (`stats.ttest_rel`) and computes Cohen’s d for
     left–right differences at each frequency.
   - Optionally generates boxplots and saves them in `results/boxplots/`.

3. **Analysis 2 – Region‑by‑region comparison**
   - For each Shen region (268 total), performs a paired t‑test across subjects
     comparing left vs right STA correlations at:
     - 5 Hz (`region_t_stats_5hz`, `region_p_values_5hz`)
     - 20 Hz (`region_t_stats_20hz`, `region_p_values_20hz`)
   - Applies:
     - Bonferroni correction (`α / 268`).
     - FDR correction (Benjamini–Hochberg), with a manual fallback if
       `false_discovery_control` is unavailable.
   - Saves regionwise statistics to CSV files in
     `results/statistics/`:
     - `region_by_region_statistics_5Hz.csv`
     - `region_by_region_statistics_20Hz.csv`

4. **Analysis 3 – Cluster permutation tests**
   - Uses `cluster_permutation_test(...)`:
     - Computes paired t‑tests left vs right at each region to obtain
       `t_stats` and `p_values`.
     - Defines an initial significant mask with uncorrected `p < 0.05`.
     - `find_clusters_3d(...)` maps region significance to 3D voxel clusters
       using the Shen atlas parcellation and returns cluster labels and sizes.
     - For `N_PERMUTATIONS` permutations:
       - Randomly flips sign of within‑subject differences (left–right vs
         right–left).
       - Runs one‑sample t‑tests across subjects on permuted differences.
       - Finds clusters in permuted data, records maximum cluster size.
     - Cluster p‑values are estimated as the proportion of permutations with
       max cluster size ≥ observed cluster size.
   - Run separately for 5 Hz and 20 Hz, with outputs saved as CSV summaries if
     `SAVE_RESULTS=True`.

5. **Visualizations**
   - **Boxplots** for average correlations at 5 Hz and 20 Hz.
   - **Cluster plots**:
     - `plot_cluster_pvalue_3x3(...)`:
       3×3 grid of axial/sagittal/coronal slices showing cluster labels.
   - **Difference maps**:
     - `plot_difference_map_3x5(...)`:
       3×5 grid of slices of the mean difference volume (left – right) mapped
       onto the Shen atlas volume.
   - **STA–correlation maps (FDR / Bonferroni corrected)**:
     - `compute_onesample_stats_with_correction(...)`:
       Performs one‑sample tests of STA–to–brain correlations vs zero and
       applies Bonferroni and FDR corrections.
     - `plot_sta_correlation_4x3(...)`:
       4×3 grids (conditions × slices) showing only significant regions for:
       - Left STA – 5 Hz
       - Right STA – 5 Hz
       - Left STA – 20 Hz
       - Right STA – 20 Hz
     - Outputs go to `results/corrected_sta_correlations/`.

6. **Summary reporting**
   - Prints a detailed text summary to the console.
   - If `SAVE_RESULTS=True`, writes a summary report to:
     - `results/statistics/analysis_summary_left_right_comparison.txt`

**Typical usage:**

```bash
cd /projectnb/viscog01/yufang/Neural_representaion_proj/STA_lateralization
python comparison_left_right.py
```

Run this **after** `get_envelope_sta_shen.py` has produced all
`*_sta_label_envelope_corr_*Hz.npy` files for the subjects of interest.

---

## Dependencies and environment

All scripts are standard Python modules and are typically run from the command
line as shown above. Required packages include (non‑exhaustive):

- `mne`
- `mne_connectivity`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `nibabel`
- `nilearn`
- `tqdm`
- `joblib`

Many steps use multiprocessing (`multiprocessing.Pool`) and set
`OMP_NUM_THREADS` via `num_threads(...)`. When running on an HPC cluster, make
sure that the number of Python processes and threads per process are compatible
with your resource allocation.


