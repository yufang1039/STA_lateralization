# STA_lateralization

Code supporting analysis of MEG source time series at subcortical STA coordinates, Shen-atlas label time courses, and left vs right hemispheric comparisons.

## File Descriptions

The repository is organized as follows:

1. `MEG_pipeline/`: Scripts for MEG source estimation, anatomical preparation, and envelope correlation analysis.
   - `regis_corr_vol.py`: Uses ANTs to compute and apply transforms (affine + warp) for volumetric correlation maps.
   - `run_freesurfer.py`: Runs `recon-all` to generate FreeSurfer outputs (including `aparc.a2009s+aseg.mgz`) for each subject.
   - `ants.sh`: Uses ANTs transforms to bring the Shen atlas into each subject’s FreeSurfer space (subject-space Shen parcellation).
   - `demo_MNI_source_estimation_freq.py`: Computes LCMV source time series at STA MNI coordinates.
   - `demo_shen_atlas_source_estimation_freq.py`: Extracts Shen-268 atlas label time courses using LCMV beamforming.
   - `demo_get_envelope_sta_shen.py`: Computes orthogonalized log-envelope correlations between STA and Shen label time series.

2. `Figure 5/`: Scripts and data for reproducing Figure 5 (Surface plotting).
   - `data/`: Contains parcellation files and statistical results (CSV).
   - `results/`: Output directory for plots.
   - `script/Figure_5C_plot_shen_surface.py`: Generates surface visualizations of significant STA–brain correlations.

## Dependencies

The code is compatible with **Linux**, **macOS**, and **Windows**. It has been tested primarily on **Linux** with **Python 3.8+**.

Please ensure the following Python libraries are installed. Recommended versions are provided below (tested environment), but the code should be compatible with recent stable releases.

*   **Python**: 3.8 or higher
*   **mne**: >= 1.0.0
*   **mne-connectivity**: >= 0.3.0
*   **numpy**: >= 1.20.0
*   **scipy**: >= 1.7.0
*   **pandas**: >= 1.3.0
*   **matplotlib**: >= 3.4.0
*   **seaborn**: >= 0.11.0
*   **nibabel**: >= 3.2.0
*   **nilearn**: >= 0.8.0
*   **joblib**: >= 1.0.0
*   **tqdm**: >= 4.60.0

In addition, the full MEG pipeline assumes you have:

*   **FreeSurfer**: installed and configured (with `SUBJECTS_DIR` and `FREESURFER_HOME`), for running `run_freesurfer.py`.
*   **ANTs** (Advanced Normalization Tools): installed and on your `PATH`, for running `regis_corr_vol.py` and `ants.sh`.

## Installation

1.  Clone this repository:
    ```bash
    git clone <repository_url>
    cd STA_lateralization
    ```

2.  Install the required dependencies using pip:
    ```bash
    pip install mne mne-connectivity numpy scipy pandas matplotlib seaborn nibabel nilearn joblib tqdm
    ```
    *Note: Ensure you have FreeSurfer and ANTs installed and configured if you plan to run the full pipeline from scratch (e.g., `SUBJECTS_DIR` setup for FreeSurfer and ANTs on your `PATH`).*

## Data Setup

To run the software on your data (or the demo subject), you must organize the files correctly.

### 1. Raw Data (for MEG Pipeline)
To run the scripts in `MEG_pipeline/`:
1.  **Create a `data` directory** in the root `STA_lateralization/` folder:
    ```bash
    mkdir data
    ```
2.  **Download the subject data** from the external link: [https://osf.io/search](https://osf.io/search) (or your specific data source).
3.  **Place the subject directory** inside `STA_lateralization/data/`.
    *   Example structure:
        ```text
        STA_lateralization/
        ├── data/
        │   └── sub-CC110033/  <-- Subject directory containing 'mri', 'meg', etc.
        ├── MEG_pipeline/
        └── ...
        ```
    *   *Note: The scripts currently look for `../data`. Ensure your directory structure matches.*

### 2. Figure Data (for Figure 5)
The data required for plotting Figure 5 is already included in `Figure 5/data/`. This contains **preprocessed analyzed data** (parcellation file and statistical results CSV) derived from the full pipeline, enabling you to reproduce the figures directly.

## Instructions Guide

### Reproducing Figure 5

To visualize the results as shown in Figure 5C:

1.  Navigate to the script directory:
    ```bash
    cd "Figure 5/script"
    ```
2.  Run the plotting script:
    ```bash
    python Figure_5C_plot_shen_surface.py
    ```
    *   The script uses the pre-computed CSV in `../data`.
    *   Plots will be saved to `Figure 5/results/`.


### Running the Analysis Pipeline (MEG_pipeline)

These scripts process raw MEG data to estimate source time series and correlations. A typical end-to-end workflow before source estimation is:

1.  **Compute ANTs registration / affine transforms (`regis_corr_vol.py`):**
    Use `regis_corr_vol.py` to run ANTs registration and obtain the transforms (affine + warp) needed for later volumetric alignment.
    ```bash
    cd MEG_pipeline
    python regis_corr_vol.py
    ```

2.  **Run FreeSurfer recon-all (`run_freesurfer.py`):**
    Generate FreeSurfer surfaces and segmentations for all subjects (including `aparc.a2009s+aseg.mgz`).
    ```bash
    cd MEG_pipeline
    python run_freesurfer.py
    ```

3.  **Warp Shen atlas into subject space (`ants.sh`):**
    Use ANTs to bring the Shen atlas into each subject’s FreeSurfer space, producing a subject-space Shen parcellation.
    ```bash
    cd MEG_pipeline
    bash ants.sh
    ```

4.  **Source Estimation (STA Coordinates):**
    Run `demo_MNI_source_estimation_freq.py` to extract time series from specific MNI coordinates.
    ```bash
    cd MEG_pipeline
    python demo_MNI_source_estimation_freq.py
    ```

5.  **Source Estimation (Shen Atlas):**
    Run `demo_shen_atlas_source_estimation_freq.py` to extract time series for the Shen atlas parcels.
    ```bash
    cd MEG_pipeline
    python demo_shen_atlas_source_estimation_freq.py
    ```

6.  **Envelope Correlation:**
    Run `demo_get_envelope_sta_shen.py` to compute the correlation between STA and atlas time series.
    ```bash
    cd MEG_pipeline
    python demo_get_envelope_sta_shen.py
    ```
    *   Outputs will be saved to `STA_lateralization/results/`.
    *   *Note: You may need to update the `case_list` variable in the scripts to match your subject IDs.*
