import mne
from mne.beamformer import make_lcmv, apply_lcmv_raw, apply_lcmv_epochs
import numpy as np
from scipy.io import savemat
import os
import sys
import os.path as op
import subprocess
from mne.transforms import apply_trans
import nibabel as nib
import multiprocessing as mp
from pathlib import Path
import subprocess
import pathlib
from mne.minimum_norm import make_inverse_operator
from mne_connectivity import envelope_correlation
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from functools import wraps
import matplotlib.pyplot as plt
import time
import pickle
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd, apply_dics,apply_dics_epochs
import itertools
import joblib

def convert(seconds): 
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print (f'@timefn: {fn.__name__} took {convert(t2-t1)} (hh:mm:ss)')
        return result
    return measure_time

def compute_SourceSpace(subject, subjects_dir, src_fname, plot=False, ss='volume', 
                        volume_spacing=7.8):
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
    conductivity = (0.3,) 
    model = mne.make_bem_model(subject=subject, ico=4,
                            conductivity=conductivity,
                            subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(fname_meg, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=5.06)
    print(fwd)
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=None)


def run_label_time_course(raw_MEG, cov, fwd, subjects_dir, subject, label_map, freq):
    global jobN4indv

    outDIR = Path('../results')        
    if not os.path.exists(outDIR):
        os.makedirs(outDIR)

    label_tc_file = f'{outDIR}/shen_label_tc_lcmv_{freq}Hz.npy'

    data_cov = mne.compute_raw_covariance(raw_MEG, n_jobs=jobN4indv)

    filters = make_lcmv(raw_MEG.info, fwd, data_cov, 0.05, cov,
                        pick_ori='max-power', weight_norm='nai')
    raw_MEG = raw_MEG.apply_hilbert(n_jobs=jobN4indv)

    stcs = apply_lcmv_raw(raw_MEG, filters, verbose=False)

    print('Extracting label time course...')
    atlas = os.path.join(subjects_dir, 'shen_1mm_268_parcellation.nii.gz')
    
    atlas = os.path.join(subjects_dir, subject, 'mri', 'shen_freesurfer.mgz')

    label_stc = mne.extract_label_time_course(stcs, (atlas, label_map), fwd['src'], mode='auto', return_generator=False, verbose=True, mri_resolution=True)

    # Save the label time course
    np.save(label_tc_file, label_stc)
    print(f'Label time course saved to {label_tc_file}')

    del raw_MEG, stcs, label_stc, filters, fwd


def num_threads(nt):
    nt = str(nt)
    os.environ["OMP_NUM_THREADS"] = nt

def run_subject_in_parallel(subjects_dir, subject, volume_spacing, label_map, freq):
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
        print (f'{trans} File doesnt exist...')

    info = mne.io.read_info(fname_meg)

    print(file_ss)
    if not file_ss.exists():

        src = compute_SourceSpace(subject, subjects_dir, src_fname, plot=True, ss='volume', 
                            volume_spacing=volume_spacing)

        src.save(src_fname, overwrite=False)
    src = mne.read_source_spaces(src_fname)

    if not file_fm.exists():
        forward_model(subject, subjects_dir, fname_meg, trans, src, fwd_fname)
    fwd = mne.read_forward_solution(fwd_fname)

       
    # sensitivty_plot(subject, subjects_dir, fwd)
    raw = mne.io.read_raw_fif(fname_meg, verbose='error', preload=True)

    srate = raw.info['sfreq']
    n_time_samps = raw.n_times
    time_secs = raw.times
    ch_names = raw.ch_names
    n_chan = len(ch_names)
    freq_res =  srate/n_time_samps
    print('\n')
    print('-------------------------- Data summary-------------------------------')
    print(f'Subject {subject}')
    print(f"Frequency resolution {freq_res} Hz")
    print(f"The first few channel names are {ch_names[:3]}")
    print(f"The last time sample at {time_secs[-1]} seconds.")
    print(f"Sampling Frequency (No of time points/sec) {srate} Hz")
    print(f"Miscellaneous acquisition info {raw.info['description']}")
    print(f"Bad channels marked during data acquisition {raw.info['bads']}")
    print(f"Convert time in sec ( 60s ) to ingeter index {raw.time_as_index(60)}") # Convert time to indices
    print(f"The raw data object has {n_time_samps} time samples and {n_chan} channels.")
    print('------------------------------------------------------------------------')
    print('\n')
    if not file_proj.exists():
        projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2, ch_name='ECG063')
        projs_eog1, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='EOG061')
        projs_eog2, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='EOG062')
        if projs_ecg is not None:
            mne.write_proj(heartbeat_proj, projs_ecg) # Saving projectors
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

    cov = mne.compute_raw_covariance(raw_proj_filtered, n_jobs=jobN4indv)

    l_freq = freq - 2.0
    h_freq = freq + 2.0
    print(f'Band pass filter data [{l_freq}, {h_freq}]')
    raw_MEG = raw_proj_filtered.filter(l_freq=l_freq, h_freq=h_freq)

    run_label_time_course(raw_MEG, cov, fwd, subjects_dir, subject, label_map, freq)

#---------------------------------------Main Program starts here-----------------------------#

@timefn
def main():
    subjects_dir = '../data'
    case_list = ['sub-CC110033']

    b = list(np.arange(1,269))
    a = [str(val) for val in b]
    label_map = dict(zip(a, b))

    volume_spacing = 7.8
    global jobN4indv
    jobN4indv = 1;
    
    for subject in case_list:
        for freq in [5, 20]: 
             run_subject_in_parallel(subjects_dir, subject, volume_spacing, label_map, freq)

if __name__ == "__main__":
    import time 
    startTime = time.time()
    main()
    print('The script took {0} second !'.format(time.time() - startTime))
