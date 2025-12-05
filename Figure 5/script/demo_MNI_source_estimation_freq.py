import mne
from mne.beamformer import make_lcmv, apply_lcmv_raw
import numpy as np
from scipy.io import savemat
import os
import sys
import os.path as op
import subprocess
from mne.transforms import apply_trans, invert_transform, read_ras_mni_t
import nibabel as nib
import multiprocessing as mp
from pathlib import Path
import pathlib
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from functools import wraps
import matplotlib.pyplot as plt
import time
import pickle
import joblib

def convert(seconds): 
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f'@timefn: {fn.__name__} took {convert(t2-t1)} (hh:mm:ss)')
        return result
    return measure_time

def MNI_to_MRI(subject, subjects_dir, t1, mni_coords):
    """
    Convert MNI coordinates to FreeSurfer MRI (surface RAS) coordinates.
    
    Parameters:
    -----------
    subject : str
        Subject ID
    subjects_dir : str
        FreeSurfer subjects directory
    t1 : nibabel image
        T1 MRI image
    mni_coords : array-like
        MNI coordinates in mm
        
    Returns:
    --------
    freesurfer_mri : array
        Coordinates in FreeSurfer MRI space (in meters)
    """
    # MNI to Native scanner RAS
    ras_mni_t = mne.transforms.read_ras_mni_t(subject, subjects_dir)
    ras_mni_t = ras_mni_t['trans']
    mni_ras_t = np.linalg.inv(ras_mni_t)
    ras_coords = apply_trans(mni_ras_t, mni_coords)
    
    # Voxel to RAS to MNI
    vox_ras_mni_t = np.dot(ras_mni_t, t1.affine)
    mni_ras_vox_t = np.linalg.inv(vox_ras_mni_t)

    VOXEL = apply_trans(mni_ras_vox_t, mni_coords)

    vox_mri_t = t1.header.get_vox2ras_tkr()
    freesurfer_mri = apply_trans(vox_mri_t, VOXEL) / 1e3

    return freesurfer_mri

def compute_SourceSpace(subject, subjects_dir, src_fname, plot=False, ss='volume', 
                          volume_spacing=7.8):
    # 7.8 as the default spacing
    src = None
    if ss == 'surface':
        src = mne.setup_source_space(subject, spacing='ico5', add_dist=None,
                                     subjects_dir=subjects_dir)
        src.save(src_fname, overwrite=False)
        if plot:
            mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                             src=src, orientation='coronal')
    elif ss == 'volume':
        surface = op.join(subjects_dir, subject, 'bem', 'inner_skull.surf')
        src = mne.setup_volume_source_space(subject, subjects_dir=subjects_dir,
                                            pos=volume_spacing, surface=surface, verbose=True)
        src.save(src_fname, overwrite=False)
        if plot:
            fig = mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                                   brain_surfaces='white', src=src, orientation='coronal', show=True)
            plt.close()
    return src

def forward_model(subject, subjects_dir, fname_meg, trans, src, fwd_fname):
    conductivity = (0.3,)  # one-layer model conductivity.
    model = mne.make_bem_model(subject=subject, ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(fname_meg, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=5.06)
    print(fwd)
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=None)
    
def run_coordinate_time_series(raw_MEG, cov, fwd, subjects_dir, subject, freq):
    """
    Computes LCMV beamformer filters, applies them to the data,
    and extracts source time series for the two target MNI coordinates.
    """
    # Create output directory
    outDIR = Path('../results')
    if not os.path.exists(outDIR):
        os.makedirs(outDIR)

    # Define MNI coordinates for the two STAs (in mm).
    right_STA_mni = np.array([18, 9, 18])
    left_STA_mni  = np.array([-6, 9, -9])

    # Load T1 image for coordinate transformation
    t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
    t1 = nib.load(t1_fname)

    # Convert MNI coordinates to FreeSurfer MRI space (surface RAS) using MNI_to_MRI
    right_STA_mri = MNI_to_MRI(subject, subjects_dir, t1, right_STA_mni)
    left_STA_mri = MNI_to_MRI(subject, subjects_dir, t1, left_STA_mni)
    
    print(f"Right STA MNI: {right_STA_mni} mm -> MRI: {right_STA_mri * 1000} mm")
    print(f"Left STA MNI: {left_STA_mni} mm -> MRI: {left_STA_mri * 1000} mm")

    # Compute data covariance if needed.
    data_cov = mne.compute_raw_covariance(raw_MEG)

    # Compute beamformer filters using LCMV.
    filters = make_lcmv(raw_MEG.info, fwd, data_cov, reg=0.05, noise_cov=cov,
                        pick_ori='max-power', weight_norm='nai')
    
    # Optionally apply Hilbert transform.
    raw_MEG.apply_hilbert()
    
    # Apply beamformer to get source estimates.
    stc = apply_lcmv_raw(raw_MEG, filters, verbose=False)
    
    # Get source space from the forward model
    src = fwd['src']

    # For VOLUME source spaces, collect all active source coordinates
    all_source_coords = []
    
    for hemi in src:
        # Get coordinates of active source points (in meters, surface RAS)
        rr = hemi['rr'][hemi['inuse'].astype(bool)]
        all_source_coords.append(rr)
    
    # Concatenate all source coordinates (already in FreeSurfer MRI space, in meters)
    all_source_coords = np.vstack(all_source_coords)  # Shape: (n_sources, 3), in meters

    # Find nearest source points to our target MRI coordinates
    def find_nearest_index(coord, coord_list):
        dists = np.linalg.norm(coord_list - coord, axis=1)
        return np.argmin(dists), np.min(dists)

    # Find best matching index (convert distance to mm for display)
    idx_right, dist_right = find_nearest_index(right_STA_mri, all_source_coords)
    idx_left, dist_left = find_nearest_index(left_STA_mri, all_source_coords)
    
    print(f"Right STA: nearest grid index {idx_right}, MRI coord: {all_source_coords[idx_right] * 1000} mm (distance: {dist_right * 1000:.2f} mm)")
    print(f"Left STA: nearest grid index {idx_left}, MRI coord: {all_source_coords[idx_left] * 1000} mm (distance: {dist_left * 1000:.2f} mm)")
    
    # Extract time series from the stc data (stc.data shape: n_points x n_times).
    ts_right = stc.data[idx_right, :]
    ts_left  = stc.data[idx_left, :]
    
    # Save the time series separately.
    right_file = op.join(outDIR, f'right_STA_ts_lcmv_{freq}Hz.npy')
    left_file  = op.join(outDIR, f'left_STA_ts_lcmv_{freq}Hz.npy')

    print("Finishing extracting, ready to save time series!")

    np.save(right_file, ts_right)
    np.save(left_file, ts_left)
    
    print(f'Right STA time series saved to {right_file}')
    print(f'Left STA time series saved to {left_file}')
    
    # Clean up.
    del raw_MEG, stc, filters

def num_threads(nt):
    os.environ["OMP_NUM_THREADS"] = str(nt)

def run_subject_in_parallel(subjects_dir, subject, volume_spacing, freq):
    global jobN4indv
    num_threads(10)
    DATA_DIR = Path(f'{subjects_dir}', f'{subject}', 'mne_files')
    eye_proj1 = f'{DATA_DIR}/{subject}_eyes1-proj.fif.gz'
    eye_proj2 = f'{DATA_DIR}/{subject}_eyes2-proj.fif.gz'
    fname_meg = f'{DATA_DIR}/{subject}_ses-rest_task-rest.fif'
    t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
    heartbeat_proj = f'{DATA_DIR}/{subject}_heartbeat-proj.fif.gz'
    fwd_fname = f'{DATA_DIR}/{subject}_{volume_spacing}-fwd.fif.gz'
    src_fname = f'{DATA_DIR}/{subject}_{volume_spacing}-src.fif.gz'
    cov_fname = f'{DATA_DIR}/{subject}-cov_{volume_spacing}.fif.gz'
    raw_cov_fname = f'{DATA_DIR}/{subject}-rawcov_{volume_spacing}.fif.gz'
    raw_proj = f'{DATA_DIR}/{subject}_ses-rest_task-rest_proj.fif.gz'

    trans = os.path.join(subjects_dir, subject, 'mne_files', f'{subject}-trans.fif')
    
    file_trans = pathlib.Path(trans)
    file_ss = pathlib.Path(src_fname)
    file_fm = pathlib.Path(fwd_fname)
    file_proj = pathlib.Path(raw_proj)
    file_cov = pathlib.Path(cov_fname)
    file_rawcov = pathlib.Path(raw_cov_fname)

    if not file_trans.exists():
        print(f'{trans} File doesn\'t exist...')
        # sys.exit(0) # Don't exit, just print warning for demo

    # info = mne.io.read_info(fname_meg)
    print(file_ss)
    if not file_ss.exists():
        src = compute_SourceSpace(subject, subjects_dir, src_fname, plot=True, ss='volume', 
                                  volume_spacing=volume_spacing)
        src.save(src_fname, overwrite=False)
    src = mne.read_source_spaces(src_fname)

    if not file_fm.exists():
        forward_model(subject, subjects_dir, fname_meg, trans, src, fwd_fname)
    fwd = mne.read_forward_solution(fwd_fname)

    raw = mne.io.read_raw_fif(fname_meg, verbose='error', preload=True)

    srate = raw.info['sfreq']
    n_time_samps = raw.n_times
    time_secs = raw.times
    ch_names = raw.ch_names
    n_chan = len(ch_names)
    freq_res = srate / n_time_samps
    print('\n')
    print('-------------------------- Data summary-------------------------------')
    print(f'Subject {subject}')
    print(f"Frequency resolution {freq_res} Hz")
    print(f"The first few channel names are {ch_names[:3]}")
    print(f"The last time sample at {time_secs[-1]} seconds.")
    print(f"Sampling Frequency {srate} Hz")
    print(f"Acquisition info: {raw.info['description']}")
    print(f"Bad channels: {raw.info['bads']}")
    print(f"Time index for 60 seconds: {raw.time_as_index(60)}")
    print(f"The raw data object has {n_time_samps} time samples and {n_chan} channels.")
    print('------------------------------------------------------------------------\n')
    
    if not file_proj.exists():
        projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2, ch_name='ECG063')
        projs_eog1, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='EOG061')
        projs_eog2, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='EOG062')
        if projs_ecg is not None:
            mne.write_proj(heartbeat_proj, projs_ecg)
            raw.info['projs'] += projs_ecg
        if projs_eog1 is not None:
            mne.write_proj(eye_proj1, projs_eog1)
            raw.info['projs'] += projs_eog1
        if projs_eog2 is not None:
            mne.write_proj(eye_proj2, projs_eog2)
            raw.info['projs'] += projs_eog2
        raw.apply_proj()
        raw.save(raw_proj, proj=True, overwrite=False)
    print(raw_proj)
    raw_proj_applied = mne.io.read_raw_fif(raw_proj, verbose='error', preload=True)

    print('High-pass filtering data at 0.5 Hz')
    raw_proj_filtered = raw_proj_applied.filter(l_freq=0.5, h_freq=None, method='iir')

    cov = mne.compute_raw_covariance(raw_proj_filtered)

    l_freq = freq - 2.0
    h_freq = freq + 2.0
    print(f'Band pass filtering data between [{l_freq}, {h_freq}] Hz')
    raw_MEG = raw_proj_filtered.filter(l_freq=l_freq, h_freq=h_freq)

    # Extract time series for the two coordinates.
    run_coordinate_time_series(raw_MEG, cov, fwd, subjects_dir, subject, freq)

#---------------------------------------Main Program starts here-----------------------------#

@timefn
def main():
    volume_spacing = 7.8
    global jobN4indv
    jobN4indv = 1
    subjects_dir = '../data'
    case_list = ['sub-CC110033']
    
    for subject in case_list:
        for freq in [5, 20]:
            print(f"Processing {subject} at {freq}Hz")
            run_subject_in_parallel(subjects_dir, subject, volume_spacing, freq)

if __name__ == "__main__":
    startTime = time.time()
    main()
    print('The script took {0} seconds!'.format(time.time() - startTime))
