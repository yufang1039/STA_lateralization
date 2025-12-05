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
        Base directory containing subject data
    subject : str
        Subject identifier
    freq : int
        Frequency band (5 or 20 Hz)
    side : str
        Hemisphere side ('left' or 'right')
    """
    # Construct file paths
    sta_file = os.path.join(subjects_dir, subject, 'mri', 'sta_time_series', 
                            f'{side}_STA_ts_lcmv_{freq}Hz.npy')
    label_file = os.path.join(subjects_dir, subject, 'mri', 'shen_label_time_course', 
                              f'shen_label_tc_lcmv_{freq}Hz.npy')
    
    # Create output directory and file path
    out_dir = os.path.join(subjects_dir, subject, 'mri', 'shen_label_envelope_corr')
    out_file = os.path.join(out_dir, f'{side}_sta_label_envelope_corr_{freq}Hz.npy')
    
    # Skip if output file already exists
    if os.path.exists(out_file):
        print(f"Output file already exists for {subject} {side} at {freq}Hz, skipping...")
        return
    
    # Skip if input files don't exist
    if not (os.path.exists(sta_file) and os.path.exists(label_file)):
        print(f"Missing input files for {subject} {side} at {freq}Hz, skipping...")
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
        # STA will be the first row (index 0)
        combined_ts = np.vstack([sta_ts, label_ts])
        
        # Wrap data in a list as expected by envelope_correlation
        combined_ts = [combined_ts]
        
        # Calculate envelope correlations for all time series
        corr_matrix = envelope_correlation(combined_ts, orthogonalize="pairwise", 
                                          log=True, absolute=True, verbose=False)
        
        # Extract only the correlations between STA (index 0) and all label time series
        corr_data = corr_matrix.get_data()
        sta_label_corr = corr_data[0, 1:]  # Shape should be (n_labels,)
        
        # Create output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        
        # Save correlation results
        np.save(out_file, sta_label_corr)
        
        print(f"Completed {subject} {side} at {freq}Hz: correlation shape {sta_label_corr.shape}")
        
    except Exception as e:
        print(f"Error processing {subject} {side} at {freq}Hz: {str(e)}")
        return

def process_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing."""
    subjects_dir, subject, freq, side = args
    try:
        return process_subject_freq_side(subjects_dir, subject, freq, side)
    except Exception as e:
        print(f"Fatal error in process_wrapper for {subject} {side} at {freq}Hz: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_subjects_with_incomplete_files(subjects_dir, cases):
    """
    Check which subjects have less than 4 files in shen_label_envelope_corr directory.
    
    Parameters:
    -----------
    subjects_dir : str
        Base directory containing subject data
    cases : str
        Path to file containing list of subjects
    
    Returns:
    --------
    list : List of subjects with less than 4 files
    """
    import os
    
    # Read case list
    with open(cases) as f:
        case_list = f.read().splitlines()
    
    subjects_with_files = 0
    subjects_without_files = 0
    total_files = 0
    subjects_with_less_than_4_files = []
    
    for subject in case_list:
        target_dir = os.path.join(subjects_dir, subject, 'mri', 'shen_label_envelope_corr')
        
        if os.path.exists(target_dir):
            # Count files in the directory
            files = os.listdir(target_dir)
            file_count = len([f for f in files if os.path.isfile(os.path.join(target_dir, f))])
            
            if file_count > 0:
                subjects_with_files += 1
                total_files += file_count
                print(f"{subject}: {file_count} files")
                
                # Track subjects with less than 4 files
                if file_count < 4:
                    subjects_with_less_than_4_files.append(subject)
            else:
                subjects_without_files += 1
                subjects_with_less_than_4_files.append(subject)
        else:
            subjects_without_files += 1
            subjects_with_less_than_4_files.append(subject)
    
    print(f"\nSummary:")
    print(f"Total subjects: {len(case_list)}")
    print(f"Subjects with files: {subjects_with_files}")
    print(f"Subjects without files (or directory not found): {subjects_without_files}")
    print(f"Total files found: {total_files}")
    print(f"Subjects with less than 4 files: {len(subjects_with_less_than_4_files)}")
    
    return subjects_with_less_than_4_files

@timefn
def main():
    # Configuration
    cases = '/projectnb/viscog01/yufang/camcan/cc700/freesurfer_output/full.txt'
    subjects_dir = '/projectnb/viscog01/yufang/camcan/cc700/freesurfer_output'

    subjects_with_less_than_4_files = get_subjects_with_incomplete_files(subjects_dir, cases)
    case_list = subjects_with_less_than_4_files
    
    # Read case list
    with open(cases) as f:
        case_list = f.read().splitlines()
    
    # Optional: Process only a portion of the case list
    # Divide the case list into 6 equal portions and select the i-th portion
    i = 1  # Change this value from 1 to 6 to select different portions
    total_portions = 6
    portion_size = len(case_list) // total_portions
    remainder = len(case_list) % total_portions
    
    # Calculate start and end indices for the i-th portion
    start_idx = (i - 1) * portion_size + min(i - 1, remainder)
    end_idx = i * portion_size + min(i, remainder)
    
    # Comment out the next line to process all subjects
    # case_list = case_list[start_idx:end_idx]
    
    # Set thread count for each process
    num_threads(10)
    
    # Create a list of all tasks (subjects_dir, subject, freq, side combinations)
    tasks = []
    for freq in [5, 20]:
        for subject in case_list:
            for side in ['left', 'right']:
                # Check if output already exists before adding to task list
                out_file = os.path.join(subjects_dir, subject, 'mri', 
                                       'shen_label_envelope_corr',
                                       f'{side}_sta_label_envelope_corr_{freq}Hz.npy')
                
                # Check if input files exist before adding to task list
                sta_file = os.path.join(subjects_dir, subject, 'mri', 'sta_time_series', 
                                       f'{side}_STA_ts_lcmv_{freq}Hz.npy')
                label_file = os.path.join(subjects_dir, subject, 'mri', 'shen_label_time_course', 
                                         f'shen_label_tc_lcmv_{freq}Hz.npy')
                
                # Only add task if output doesn't exist AND both input files exist
                if not os.path.exists(out_file) and os.path.exists(sta_file) and os.path.exists(label_file):
                    tasks.append((subjects_dir, subject, freq, side))
    
    print(f"Total tasks to process: {len(tasks)}")
    
    # Create a pool of workers
    num_processes = min(mp.cpu_count(), 8)  # Use up to 8 processes
    print(f"Starting pool with {num_processes} processes")
    
    # Execute tasks in parallel
    pool = mp.Pool(processes=num_processes)
    pool.map(process_wrapper, tasks)
    pool.close()
    pool.join()

if __name__ == "__main__":
    startTime = time.time()
    main()
    print('The script took {0} seconds!'.format(time.time() - startTime))