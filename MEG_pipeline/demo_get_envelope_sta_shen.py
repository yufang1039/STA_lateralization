import os
import numpy as np
from pathlib import Path
from mne_connectivity import envelope_correlation
import multiprocessing as mp
import time
from functools import wraps

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

def num_threads(nt):
    nt = str(nt)
    os.environ["OMP_NUM_THREADS"] = nt

def process_subject_freq_side(subjects_dir, subject, freq, side):
    """
    Process envelope correlation for a single subject, frequency, and hemisphere side.
    
    Parameters:
    -----------
    subjects_dir : str
        Base directory containing subject data (used for logging/context, though inputs are now in results)
    subject : str
        Subject identifier
    freq : int
        Frequency band (5 or 20 Hz)
    side : str
        Hemisphere side ('left' or 'right')
    """
    # Define input/output directories
    results_dir = '../results'
    
    # Construct file paths
    sta_file = os.path.join(results_dir, f'{side}_STA_ts_lcmv_{freq}Hz.npy')
    label_file = os.path.join(results_dir, f'shen_label_tc_lcmv_{freq}Hz.npy')
    
    # Create output directory if needed (results_dir should exist)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    out_file = os.path.join(results_dir, f'{side}_sta_label_envelope_corr_{freq}Hz.npy')
    
    # Skip if input files don't exist
    if not (os.path.exists(sta_file) and os.path.exists(label_file)):
        print(f"Missing input files for {subject} {side} at {freq}Hz in {results_dir}, skipping...")
        return

    print(f"Processing {subject} {side} at {freq}Hz...")
        
    try:
        # Load data
        sta_ts = np.load(sta_file)
        label_ts = np.load(label_file)
        
        # Reshape STA time series to match expected input format
        sta_ts = sta_ts.reshape(1, -1)  # Make it 2D: 1 signal x timepoints
        
        # Ensure label_ts is 2D (regions x timepoints)
        if len(label_ts.shape) == 1:
            label_ts = label_ts.reshape(1, -1)
        
        # Concatenate STA time series with label time series
        combined_ts = np.vstack([sta_ts, label_ts])
        
        # Wrap data in a list as expected by envelope_correlation
        combined_ts = [combined_ts]
        
        # Calculate envelope correlations for all time series
        combined_ts_float = [c.astype(np.float64) for c in combined_ts] # Ensure float for connectivity

        # Calculate envelope correlations for all time series
        corr_matrix = envelope_correlation(combined_ts_float, orthogonalize="pairwise", 
                                          log=True, absolute=True, verbose=False)
        
        # Extract only the correlations between STA (index 0) and all label time series
        corr_data = corr_matrix.get_data()
        sta_label_corr = corr_data[0, 1:]  
        
        # Save correlation results
        np.save(out_file, sta_label_corr)
        
        print(f"Completed {subject} {side} at {freq}Hz: correlation shape {sta_label_corr.shape}")
        
    except Exception as e:
        print(f"Error processing {subject} {side} at {freq}Hz: {str(e)}")
        import traceback
        traceback.print_exc()
        return

@timefn
def main():
    # Configuration
    subjects_dir = '../data'
    case_list = ['sub-CC110033']
    
    # Set thread count for each process
    num_threads(10)
    
    for subject in case_list:
        for freq in [5, 20]:
            for side in ['left', 'right']:
                 process_subject_freq_side(subjects_dir, subject, freq, side)

if __name__ == "__main__":
    startTime = time.time()
    main()
    print('The script took {0} seconds!'.format(time.time() - startTime))
