import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import label
from pathlib import Path
import mne
from mne.datasets import fetch_fsaverage
import nibabel as nib
from tqdm import tqdm

# Set up paths
cases = '/projectnb/viscog01/yufang/camcan/cc700/freesurfer_output/full.txt'
subjects_dir = '/projectnb/viscog01/yufang/camcan/cc700/freesurfer_output'
results_dir = '/projectnb/viscog01/yufang/Neural_representaion_proj/results'

# Configuration
SAVE_RESULTS = True  # Set to True to save results to disk
N_PERMUTATIONS = 5000  # Number of permutations for cluster test

# Subjects to exclude
excluded_subjects = {
    'sub-CC222326',
    'sub-CC120264',
    'sub-CC120313',
    'sub-CC120462',
    'sub-CC120469',
    'sub-CC510208'
}

# Create results directory structure if it doesn't exist (only if saving)
if SAVE_RESULTS:
    os.makedirs(results_dir, exist_ok=True)
    # Create subfolders for organized outputs
    os.makedirs(os.path.join(results_dir, 'cluster_analysis'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'cluster_analysis', '5Hz'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'cluster_analysis', '20Hz'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'statistics'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'boxplots'), exist_ok=True)

# Load subject list and filter exclusions
with open(cases) as f:
    case_list = f.read().splitlines()

# Filter out excluded subjects
case_list = [s for s in case_list if s not in excluded_subjects]

base_dir = '/projectnb/viscog01/yufang/camcan/cc700/freesurfer_output'

# Initialize data storage
results_data = []
# Store individual region correlations for region-by-region analysis
region_correlations = {
    'left_5Hz': [],
    'right_5Hz': [],
    'left_20Hz': [],
    'right_20Hz': []
}

print("Processing subjects...")
print(f"Excluding {len(excluded_subjects)} subjects: {excluded_subjects}")

for subject in case_list:
    # Check if all required files exist for this subject
    corr_dir = os.path.join(base_dir, subject, 'mri', 'shen_label_envelope_corr')
    
    required_files = [
        f'left_sta_label_envelope_corr_5Hz.npy',
        f'right_sta_label_envelope_corr_5Hz.npy',
        f'left_sta_label_envelope_corr_20Hz.npy',
        f'right_sta_label_envelope_corr_20Hz.npy'
    ]
    
    # Check if all files exist
    all_files_exist = all(os.path.exists(os.path.join(corr_dir, f)) for f in required_files)
    
    if not all_files_exist:
        print(f"Missing files for {subject}, skipping...")
        continue
    
    print(f"Processing {subject}...")
    
    # Load and process each file
    subject_data = {'subject': subject}
    
    # Load left hemisphere 5Hz
    left_5hz_path = os.path.join(corr_dir, 'left_sta_label_envelope_corr_5Hz.npy')
    left_5hz_matrix = np.load(left_5hz_path)
    
    # Load right hemisphere 5Hz
    right_5hz_path = os.path.join(corr_dir, 'right_sta_label_envelope_corr_5Hz.npy')
    right_5hz_matrix = np.load(right_5hz_path)
    
    # Load left hemisphere 20Hz
    left_20hz_path = os.path.join(corr_dir, 'left_sta_label_envelope_corr_20Hz.npy')
    left_20hz_matrix = np.load(left_20hz_path)
    
    # Load right hemisphere 20Hz
    right_20hz_path = os.path.join(corr_dir, 'right_sta_label_envelope_corr_20Hz.npy')
    right_20hz_matrix = np.load(right_20hz_path)
    
    # Squeeze out any singleton dimensions
    left_5hz_matrix = np.squeeze(left_5hz_matrix)
    right_5hz_matrix = np.squeeze(right_5hz_matrix)
    left_20hz_matrix = np.squeeze(left_20hz_matrix)
    right_20hz_matrix = np.squeeze(right_20hz_matrix)
    
    print(f"  Left 5Hz: matrix shape {left_5hz_matrix.shape}")
    print(f"  Right 5Hz: matrix shape {right_5hz_matrix.shape}")
    print(f"  Left 20Hz: matrix shape {left_20hz_matrix.shape}")
    print(f"  Right 20Hz: matrix shape {right_20hz_matrix.shape}")
    
    # Extract correlations between STA and brain regions
    # The matrix shape is (268, 269) where:
    # - 268 rows correspond to Shen regions
    # - 269 columns: first column (index 0) is STA, remaining 268 columns are Shen regions
    # We want the correlation between STA (column 0) and each of the 268 brain regions (rows)
    left_5hz_sta_brain = left_5hz_matrix[:, 0]  # 268 correlations (STA column for all regions)
    right_5hz_sta_brain = right_5hz_matrix[:, 0]  # 268 correlations
    left_20hz_sta_brain = left_20hz_matrix[:, 0]  # 268 correlations
    right_20hz_sta_brain = right_20hz_matrix[:, 0]  # 268 correlations
    
    print(f"    Left 5Hz STA-brain correlations: shape {left_5hz_sta_brain.shape}")
    print(f"    Right 5Hz STA-brain correlations: shape {right_5hz_sta_brain.shape}")
    print(f"    Left 20Hz STA-brain correlations: shape {left_20hz_sta_brain.shape}")
    print(f"    Right 20Hz STA-brain correlations: shape {right_20hz_sta_brain.shape}")
    # Store individual region correlations for region-by-region analysis
    region_correlations['left_5Hz'].append(left_5hz_sta_brain)
    region_correlations['right_5Hz'].append(right_5hz_sta_brain)
    region_correlations['left_20Hz'].append(left_20hz_sta_brain)
    region_correlations['right_20Hz'].append(right_20hz_sta_brain)
    
    # Calculate average correlation across all 268 regions
    left_5hz_avg = np.mean(left_5hz_sta_brain)
    right_5hz_avg = np.mean(right_5hz_sta_brain)
    left_20hz_avg = np.mean(left_20hz_sta_brain)
    right_20hz_avg = np.mean(right_20hz_sta_brain)
    
    # Store in subject data
    subject_data['left_5Hz_avg'] = left_5hz_avg
    subject_data['right_5Hz_avg'] = right_5hz_avg
    subject_data['left_20Hz_avg'] = left_20hz_avg
    subject_data['right_20Hz_avg'] = right_20hz_avg
    subject_data['left_5Hz_std'] = np.std(left_5hz_sta_brain)
    subject_data['right_5Hz_std'] = np.std(right_5hz_sta_brain)
    subject_data['left_20Hz_std'] = np.std(left_20hz_sta_brain)
    subject_data['right_20Hz_std'] = np.std(right_20hz_sta_brain)
    subject_data['left_5Hz_median'] = np.median(left_5hz_sta_brain)
    subject_data['right_5Hz_median'] = np.median(right_5hz_sta_brain)
    subject_data['left_20Hz_median'] = np.median(left_20hz_sta_brain)
    subject_data['right_20Hz_median'] = np.median(right_20hz_sta_brain)
    
    results_data.append(subject_data)

# Convert to DataFrame
df = pd.DataFrame(results_data)

print(f"\nProcessed {len(df)} subjects with complete data")

# Convert region correlations to arrays
# Shape: (n_subjects, 268 regions)
left_5hz_regions = np.array(region_correlations['left_5Hz'])
right_5hz_regions = np.array(region_correlations['right_5Hz'])
left_20hz_regions = np.array(region_correlations['left_20Hz'])
right_20hz_regions = np.array(region_correlations['right_20Hz'])

print(f"\nRegion correlation arrays shape: {left_5hz_regions.shape}")

# Save CSV (optional)
if SAVE_RESULTS:
    csv_path = os.path.join(results_dir, 'statistics', 'sta_brain_correlations_left_right_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

# ============================================================================
# CLUSTER PERMUTATION TEST FUNCTIONS
# ============================================================================

def find_clusters_3d(atlas_data, significant_mask, atlas_labels):
    """
    Find spatial clusters in 3D brain volume
    
    Parameters:
    -----------
    atlas_data : ndarray
        3D atlas volume with region labels
    significant_mask : array
        Boolean mask indicating significant regions
    atlas_labels : array
        Array of region labels (1-indexed)
    
    Returns:
    --------
    cluster_img : ndarray
        3D volume with cluster labels
    cluster_sizes : array
        Size of each cluster in voxels
    """
    # Create binary mask in 3D space
    binary_mask = np.zeros_like(atlas_data, dtype=bool)
    
    for idx, is_sig in enumerate(significant_mask):
        if is_sig:
            region_label = atlas_labels[idx]
            binary_mask[atlas_data == region_label] = True
    
    # Find connected components (clusters)
    cluster_img, n_clusters = label(binary_mask)
    
    # Calculate cluster sizes
    cluster_sizes = np.array([np.sum(cluster_img == i) for i in range(1, n_clusters + 1)])
    
    return cluster_img, cluster_sizes

def cluster_permutation_test(left_data, right_data, atlas_path, n_permutations=5000, random_seed=42):
    """
    Perform cluster-based permutation test for paired data
    
    Parameters:
    -----------
    left_data : ndarray
        Shape (n_subjects, n_regions) - left hemisphere data
    right_data : ndarray
        Shape (n_subjects, n_regions) - right hemisphere data
    atlas_path : str
        Path to atlas NIfTI file
    n_permutations : int
        Number of permutations
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    cluster_pvals : array
        P-value for each cluster
    cluster_img : ndarray
        3D image with cluster labels
    t_stats : array
        T-statistic for each region
    p_values : array
        Uncorrected p-value for each region
    """
    np.random.seed(random_seed)
    
    n_subjects, n_regions = left_data.shape
    
    # Load atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    atlas_labels = np.arange(1, n_regions + 1)
    
    # Step 1: Calculate observed t-statistics
    print("Calculating observed t-statistics...")
    t_stats = np.zeros(n_regions)
    p_values = np.zeros(n_regions)
    
    for region_idx in range(n_regions):
        t_stat, p_val = stats.ttest_rel(left_data[:, region_idx], right_data[:, region_idx])
        t_stats[region_idx] = t_stat
        p_values[region_idx] = p_val
    
    # Step 2: Find observed clusters (using uncorrected p < 0.05)
    print("Finding observed clusters...")
    significant_mask = p_values < 0.05
    cluster_img, cluster_sizes = find_clusters_3d(atlas_data, significant_mask, atlas_labels)
    n_clusters = len(cluster_sizes)
    
    if n_clusters == 0:
        print("No clusters found with uncorrected p < 0.05")
        return np.array([]), cluster_img, t_stats, p_values
    
    print(f"Found {n_clusters} observed clusters")
    print(f"Cluster sizes (voxels): {cluster_sizes}")
    
    # Step 3: Permutation test
    print(f"Running {n_permutations} permutations...")
    max_cluster_sizes_null = []
    
    for perm_idx in tqdm(range(n_permutations)):
        # Randomly flip sign for each subject
        sign_flips = np.random.choice([-1, 1], size=n_subjects)
        
        # Create permuted data
        perm_diff = np.zeros((n_subjects, n_regions))
        for subj_idx in range(n_subjects):
            if sign_flips[subj_idx] == 1:
                perm_diff[subj_idx] = left_data[subj_idx] - right_data[subj_idx]
            else:
                perm_diff[subj_idx] = right_data[subj_idx] - left_data[subj_idx]
        
        # Calculate t-statistics for permuted data
        perm_t_stats = np.zeros(n_regions)
        perm_p_values = np.zeros(n_regions)
        
        for region_idx in range(n_regions):
            # One-sample t-test against zero
            t_stat, p_val = stats.ttest_1samp(perm_diff[:, region_idx], 0)
            perm_t_stats[region_idx] = t_stat
            perm_p_values[region_idx] = p_val
        
        # Find clusters in permuted data
        perm_significant_mask = perm_p_values < 0.05
        _, perm_cluster_sizes = find_clusters_3d(atlas_data, perm_significant_mask, atlas_labels)
        
        # Store maximum cluster size
        if len(perm_cluster_sizes) > 0:
            max_cluster_sizes_null.append(np.max(perm_cluster_sizes))
        else:
            max_cluster_sizes_null.append(0)
    
    max_cluster_sizes_null = np.array(max_cluster_sizes_null)
    
    # Step 4: Calculate cluster p-values
    print("Calculating cluster p-values...")
    cluster_pvals = np.zeros(n_clusters)
    
    for cluster_idx in range(n_clusters):
        observed_size = cluster_sizes[cluster_idx]
        # P-value: proportion of permutations with max cluster size >= observed size
        cluster_pvals[cluster_idx] = np.mean(max_cluster_sizes_null >= observed_size)
    
    print(f"Cluster p-values: {cluster_pvals}")
    
    return cluster_pvals, cluster_img, t_stats, p_values, cluster_sizes

# ============================================================================
# ANALYSIS 1: Average correlation across all 268 regions - 5Hz comparison
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 1A: AVERAGE CORRELATION ACROSS ALL 268 REGIONS - 5Hz")
print("="*70)

left_5hz_avg = df['left_5Hz_avg'].values
right_5hz_avg = df['right_5Hz_avg'].values

print(f"\nDescriptive Statistics:")
print("-" * 50)
print(f"Left 5Hz:    Mean={np.mean(left_5hz_avg):.4f}, SD={np.std(left_5hz_avg):.4f}, "
      f"Median={np.median(left_5hz_avg):.4f}, N={len(left_5hz_avg)}")
print(f"Right 5Hz:   Mean={np.mean(right_5hz_avg):.4f}, SD={np.std(right_5hz_avg):.4f}, "
      f"Median={np.median(right_5hz_avg):.4f}, N={len(right_5hz_avg)}")

# Paired t-test
t_stat_5hz, p_val_5hz = stats.ttest_rel(left_5hz_avg, right_5hz_avg)
print(f"\nPaired T-test (Left 5Hz vs Right 5Hz):")
print(f"  t-statistic: {t_stat_5hz:.4f}")
print(f"  p-value: {p_val_5hz:.6f}")
print(f"  Significant at α=0.05: {'Yes' if p_val_5hz < 0.05 else 'No'}")

# Effect size (Cohen's d for paired samples)
diff_5hz = left_5hz_avg - right_5hz_avg
cohens_d_5hz = np.mean(diff_5hz) / np.std(diff_5hz) if np.std(diff_5hz) > 0 else 0
print(f"  Cohen's d: {cohens_d_5hz:.4f}")

# ============================================================================
# ANALYSIS 1B: Average correlation across all 268 regions - 20Hz comparison
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 1B: AVERAGE CORRELATION ACROSS ALL 268 REGIONS - 20Hz")
print("="*70)

left_20hz_avg = df['left_20Hz_avg'].values
right_20hz_avg = df['right_20Hz_avg'].values

print(f"\nDescriptive Statistics:")
print("-" * 50)
print(f"Left 20Hz:   Mean={np.mean(left_20hz_avg):.4f}, SD={np.std(left_20hz_avg):.4f}, "
      f"Median={np.median(left_20hz_avg):.4f}, N={len(left_20hz_avg)}")
print(f"Right 20Hz:  Mean={np.mean(right_20hz_avg):.4f}, SD={np.std(right_20hz_avg):.4f}, "
      f"Median={np.median(right_20hz_avg):.4f}, N={len(right_20hz_avg)}")

# Paired t-test
t_stat_20hz, p_val_20hz = stats.ttest_rel(left_20hz_avg, right_20hz_avg)
print(f"\nPaired T-test (Left 20Hz vs Right 20Hz):")
print(f"  t-statistic: {t_stat_20hz:.4f}")
print(f"  p-value: {p_val_20hz:.6f}")
print(f"  Significant at α=0.05: {'Yes' if p_val_20hz < 0.05 else 'No'}")

# Effect size (Cohen's d for paired samples)
diff_20hz = left_20hz_avg - right_20hz_avg
cohens_d_20hz = np.mean(diff_20hz) / np.std(diff_20hz) if np.std(diff_20hz) > 0 else 0
print(f"  Cohen's d: {cohens_d_20hz:.4f}")

# ============================================================================
# ANALYSIS 2A: Region-by-region comparison (268 regions) - 5Hz
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 2A: REGION-BY-REGION COMPARISON - 5Hz (268 SHEN ATLAS REGIONS)")
print("="*70)

# Perform paired t-test for each of the 268 regions
region_t_stats_5hz = []
region_p_values_5hz = []
region_cohens_d_5hz = []

# Get the actual number of regions from the data shape
n_regions = left_5hz_regions.shape[1]
print(f"\nNumber of regions in data: {n_regions}")

for region_idx in range(n_regions):
    left_5hz_region = left_5hz_regions[:, region_idx]
    right_5hz_region = right_5hz_regions[:, region_idx]
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(left_5hz_region, right_5hz_region)
    region_t_stats_5hz.append(t_stat)
    region_p_values_5hz.append(p_val)
    
    # Cohen's d
    diff = left_5hz_region - right_5hz_region
    d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
    region_cohens_d_5hz.append(d)

region_p_values_5hz = np.array(region_p_values_5hz)
region_t_stats_5hz = np.array(region_t_stats_5hz)
region_cohens_d_5hz = np.array(region_cohens_d_5hz)

# Bonferroni correction
bonferroni_alpha = 0.05 / n_regions
significant_regions_bonferroni_5hz = region_p_values_5hz < bonferroni_alpha

# FDR correction (Benjamini-Hochberg)
try:
    from scipy.stats import false_discovery_control
    fdr_corrected_5hz = false_discovery_control(region_p_values_5hz, method='bh')
    significant_regions_fdr_5hz = fdr_corrected_5hz < 0.05
    fdr_available = True
except ImportError:
    print("\nNote: false_discovery_control not available in this scipy version.")
    print("Using manual Benjamini-Hochberg FDR correction instead.")
    # Manual Benjamini-Hochberg FDR correction
    sorted_indices = np.argsort(region_p_values_5hz)
    sorted_p_values = region_p_values_5hz[sorted_indices]
    n = len(sorted_p_values)
    fdr_threshold = 0.05
    
    # Find the largest i such that P(i) <= (i/n) * alpha
    significant_mask = np.zeros(n, dtype=bool)
    for i in range(n-1, -1, -1):
        if sorted_p_values[i] <= ((i + 1) / n) * fdr_threshold:
            significant_mask[:i+1] = True
            break
    
    # Map back to original order
    significant_regions_fdr_5hz = np.zeros(n, dtype=bool)
    significant_regions_fdr_5hz[sorted_indices] = significant_mask
    fdr_available = False

print(f"\nRegion-by-region statistics (5Hz):")
print(f"  Total regions tested: {n_regions}")
print(f"  Bonferroni corrected alpha: {bonferroni_alpha:.6f}")
print(f"  Regions significant (uncorrected p < 0.05): {np.sum(region_p_values_5hz < 0.05)}")
print(f"  Regions significant (Bonferroni corrected): {np.sum(significant_regions_bonferroni_5hz)}")
print(f"  Regions significant (FDR corrected): {np.sum(significant_regions_fdr_5hz)}")

if np.sum(significant_regions_bonferroni_5hz) > 0:
    print(f"\nRegions significant after Bonferroni correction (5Hz):")
    sig_indices = np.where(significant_regions_bonferroni_5hz)[0]
    for idx in sig_indices:
        print(f"  Region {idx+1}: t={region_t_stats_5hz[idx]:.3f}, p={region_p_values_5hz[idx]:.6f}, d={region_cohens_d_5hz[idx]:.3f}")

if np.sum(significant_regions_fdr_5hz) > 0:
    print(f"\nRegions significant after FDR correction (5Hz):")
    sig_indices = np.where(significant_regions_fdr_5hz)[0]
    for idx in sig_indices[:10]:  # Show first 10
        print(f"  Region {idx+1}: t={region_t_stats_5hz[idx]:.3f}, p={region_p_values_5hz[idx]:.6f}, d={region_cohens_d_5hz[idx]:.3f}")
    if len(sig_indices) > 10:
        print(f"  ... and {len(sig_indices) - 10} more regions")

# ============================================================================
# ANALYSIS 2B: Region-by-region comparison (268 regions) - 20Hz
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 2B: REGION-BY-REGION COMPARISON - 20Hz (268 SHEN ATLAS REGIONS)")
print("="*70)

# Perform paired t-test for each of the 268 regions
region_t_stats_20hz = []
region_p_values_20hz = []
region_cohens_d_20hz = []

for region_idx in range(n_regions):
    left_20hz_region = left_20hz_regions[:, region_idx]
    right_20hz_region = right_20hz_regions[:, region_idx]
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(left_20hz_region, right_20hz_region)
    region_t_stats_20hz.append(t_stat)
    region_p_values_20hz.append(p_val)
    
    # Cohen's d
    diff = left_20hz_region - right_20hz_region
    d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
    region_cohens_d_20hz.append(d)

region_p_values_20hz = np.array(region_p_values_20hz)
region_t_stats_20hz = np.array(region_t_stats_20hz)
region_cohens_d_20hz = np.array(region_cohens_d_20hz)

# Bonferroni correction
significant_regions_bonferroni_20hz = region_p_values_20hz < bonferroni_alpha

# FDR correction (Benjamini-Hochberg)
try:
    from scipy.stats import false_discovery_control
    fdr_corrected_20hz = false_discovery_control(region_p_values_20hz, method='bh')
    significant_regions_fdr_20hz = fdr_corrected_20hz < 0.05
except ImportError:
    # Manual Benjamini-Hochberg FDR correction
    sorted_indices = np.argsort(region_p_values_20hz)
    sorted_p_values = region_p_values_20hz[sorted_indices]
    n = len(sorted_p_values)
    fdr_threshold = 0.05
    
    # Find the largest i such that P(i) <= (i/n) * alpha
    significant_mask = np.zeros(n, dtype=bool)
    for i in range(n-1, -1, -1):
        if sorted_p_values[i] <= ((i + 1) / n) * fdr_threshold:
            significant_mask[:i+1] = True
            break
    
    # Map back to original order
    significant_regions_fdr_20hz = np.zeros(n, dtype=bool)
    significant_regions_fdr_20hz[sorted_indices] = significant_mask

print(f"\nRegion-by-region statistics (20Hz):")
print(f"  Total regions tested: {n_regions}")
print(f"  Bonferroni corrected alpha: {bonferroni_alpha:.6f}")
print(f"  Regions significant (uncorrected p < 0.05): {np.sum(region_p_values_20hz < 0.05)}")
print(f"  Regions significant (Bonferroni corrected): {np.sum(significant_regions_bonferroni_20hz)}")
print(f"  Regions significant (FDR corrected): {np.sum(significant_regions_fdr_20hz)}")

if np.sum(significant_regions_bonferroni_20hz) > 0:
    print(f"\nRegions significant after Bonferroni correction (20Hz):")
    sig_indices = np.where(significant_regions_bonferroni_20hz)[0]
    for idx in sig_indices:
        print(f"  Region {idx+1}: t={region_t_stats_20hz[idx]:.3f}, p={region_p_values_20hz[idx]:.6f}, d={region_cohens_d_20hz[idx]:.3f}")

if np.sum(significant_regions_fdr_20hz) > 0:
    print(f"\nRegions significant after FDR correction (20Hz):")
    sig_indices = np.where(significant_regions_fdr_20hz)[0]
    for idx in sig_indices[:10]:  # Show first 10
        print(f"  Region {idx+1}: t={region_t_stats_20hz[idx]:.3f}, p={region_p_values_20hz[idx]:.6f}, d={region_cohens_d_20hz[idx]:.3f}")
    if len(sig_indices) > 10:
        print(f"  ... and {len(sig_indices) - 10} more regions")

# Create DataFrames with region-by-region results
region_results_5hz_df = pd.DataFrame({
    'region': np.arange(1, n_regions + 1),
    't_statistic': region_t_stats_5hz,
    'p_value': region_p_values_5hz,
    'cohens_d': region_cohens_d_5hz,
    'significant_uncorrected': region_p_values_5hz < 0.05,
    'significant_bonferroni': significant_regions_bonferroni_5hz,
    'significant_fdr': significant_regions_fdr_5hz
})

region_results_20hz_df = pd.DataFrame({
    'region': np.arange(1, n_regions + 1),
    't_statistic': region_t_stats_20hz,
    'p_value': region_p_values_20hz,
    'cohens_d': region_cohens_d_20hz,
    'significant_uncorrected': region_p_values_20hz < 0.05,
    'significant_bonferroni': significant_regions_bonferroni_20hz,
    'significant_fdr': significant_regions_fdr_20hz
})

# Save region-by-region results (optional)
if SAVE_RESULTS:
    region_stats_5hz_path = os.path.join(results_dir, 'statistics', 'region_by_region_statistics_5Hz.csv')
    region_results_5hz_df.to_csv(region_stats_5hz_path, index=False)
    print(f"\nSaved region-by-region statistics (5Hz) to {region_stats_5hz_path}")
    
    region_stats_20hz_path = os.path.join(results_dir, 'statistics', 'region_by_region_statistics_20Hz.csv')
    region_results_20hz_df.to_csv(region_stats_20hz_path, index=False)
    print(f"Saved region-by-region statistics (20Hz) to {region_stats_20hz_path}")

# ============================================================================
# CLUSTER PERMUTATION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("CLUSTER PERMUTATION ANALYSIS")
print("="*70)

atlas_path = '/projectnb/viscog01/yufang/Neural_representaion_proj/data/shen_2mm_268_parcellation.nii.gz'

# Run cluster permutation test for 5Hz
print("\n" + "="*70)
print("5Hz CLUSTER PERMUTATION TEST")
print("="*70)
cluster_pvals_5hz, cluster_img_5hz, t_stats_5hz, p_values_5hz, cluster_sizes_5hz = cluster_permutation_test(
    left_5hz_regions, right_5hz_regions, atlas_path, n_permutations=N_PERMUTATIONS
)

# Run cluster permutation test for 20Hz
print("\n" + "="*70)
print("20Hz CLUSTER PERMUTATION TEST")
print("="*70)
cluster_pvals_20hz, cluster_img_20hz, t_stats_20hz, p_values_20hz, cluster_sizes_20hz = cluster_permutation_test(
    left_20hz_regions, right_20hz_regions, atlas_path, n_permutations=N_PERMUTATIONS
)

# Save cluster results
if SAVE_RESULTS and len(cluster_pvals_5hz) > 0:
    cluster_results_5hz = pd.DataFrame({
        'cluster': np.arange(1, len(cluster_pvals_5hz) + 1),
        'size_voxels': cluster_sizes_5hz,
        'p_value': cluster_pvals_5hz,
        'significant': cluster_pvals_5hz < 0.05
    })
    cluster_results_5hz.to_csv(
        os.path.join(results_dir, 'statistics', 'cluster_results_5Hz.csv'),
        index=False
    )
    print(f"\nSaved 5Hz cluster results")

if SAVE_RESULTS and len(cluster_pvals_20hz) > 0:
    cluster_results_20hz = pd.DataFrame({
        'cluster': np.arange(1, len(cluster_pvals_20hz) + 1),
        'size_voxels': cluster_sizes_20hz,
        'p_value': cluster_pvals_20hz,
        'significant': cluster_pvals_20hz < 0.05
    })
    cluster_results_20hz.to_csv(
        os.path.join(results_dir, 'statistics', 'cluster_results_20Hz.csv'),
        index=False
    )
    print(f"\nSaved 20Hz cluster results")

# ============================================================================
# VISUALIZATION
# ============================================================================

# Create boxplot for average correlations - 5Hz
plt.figure(figsize=(10, 6))
plot_data_5hz = pd.DataFrame({
    'Condition': ['Left 5Hz'] * len(left_5hz_avg) + ['Right 5Hz'] * len(right_5hz_avg),
    'Average Correlation': np.concatenate([left_5hz_avg, right_5hz_avg])
})
sns.boxplot(data=plot_data_5hz, x='Condition', y='Average Correlation')
plt.title('STA-Brain Envelope Correlations: Left vs Right Hemisphere at 5Hz\n(Average across 268 Shen regions)', 
          fontsize=14, fontweight='bold')
plt.ylabel('Average Envelope Correlation', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

if SAVE_RESULTS:
    boxplot_5hz_path = os.path.join(results_dir, 'boxplots', 'left_right_5hz_comparison_boxplot.pdf')
    plt.savefig(boxplot_5hz_path, bbox_inches='tight')
    print(f"Saved 5Hz boxplot to {boxplot_5hz_path}")
    plt.close()
else:
    plt.show()

# Create boxplot for average correlations - 20Hz
plt.figure(figsize=(10, 6))
plot_data_20hz = pd.DataFrame({
    'Condition': ['Left 20Hz'] * len(left_20hz_avg) + ['Right 20Hz'] * len(right_20hz_avg),
    'Average Correlation': np.concatenate([left_20hz_avg, right_20hz_avg])
})
sns.boxplot(data=plot_data_20hz, x='Condition', y='Average Correlation')
plt.title('STA-Brain Envelope Correlations: Left vs Right Hemisphere at 20Hz\n(Average across 268 Shen regions)', 
          fontsize=14, fontweight='bold')
plt.ylabel('Average Envelope Correlation', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

if SAVE_RESULTS:
    boxplot_20hz_path = os.path.join(results_dir, 'boxplots', 'left_right_20hz_comparison_boxplot.pdf')
    plt.savefig(boxplot_20hz_path, bbox_inches='tight')
    print(f"Saved 20Hz boxplot to {boxplot_20hz_path}")
    plt.close()
else:
    plt.show()

# ============================================================================
# NEW VISUALIZATION FUNCTIONS
# ============================================================================

def plot_cluster_pvalue_3x3(cluster_img, atlas_img, freq_label, output_dir):
    """
    Plot 3x3 grid showing p-value clusters (3 views x 3 slices each)
    
    Parameters:
    -----------
    cluster_img : ndarray
        3D volume with cluster labels
    atlas_img : nibabel image
        Atlas image for reference
    freq_label : str
        Frequency label (e.g., '5Hz', '20Hz')
    output_dir : str
        Directory to save the figure
    """
    from nilearn import plotting
    
    # Create NIfTI image for clusters
    cluster_nii = nib.Nifti1Image(cluster_img, atlas_img.affine, atlas_img.header)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Axial slices (z-axis) - row 0
    z_coords = np.linspace(-40, 60, 3)
    for i, z in enumerate(z_coords):
        ax = axes[0, i]
        display = plotting.plot_stat_map(
            cluster_nii,
            cut_coords=[z],
            display_mode='z',
            axes=ax,
            colorbar=False,
            cmap='tab10',
            threshold=0.5,
            title=f'Axial z={z:.0f}'
        )
    
    # Sagittal slices (x-axis) - row 1
    x_coords = np.linspace(-40, 40, 3)
    for i, x in enumerate(x_coords):
        ax = axes[1, i]
        display = plotting.plot_stat_map(
            cluster_nii,
            cut_coords=[x],
            display_mode='x',
            axes=ax,
            colorbar=False,
            cmap='tab10',
            threshold=0.5,
            title=f'Sagittal x={x:.0f}'
        )
    
    # Coronal slices (y-axis) - row 2
    y_coords = np.linspace(-60, 20, 3)
    for i, y in enumerate(y_coords):
        ax = axes[2, i]
        display = plotting.plot_stat_map(
            cluster_nii,
            cut_coords=[y],
            display_mode='y',
            axes=ax,
            colorbar=False,
            cmap='tab10',
            threshold=0.5,
            title=f'Coronal y={y:.0f}'
        )
    
    plt.suptitle(f'Significant Clusters - {freq_label}\n(Cluster Permutation Test, p < 0.05)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'clusters_3x3_{freq_label}.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved 3x3 cluster plot to {output_path}")

def plot_difference_map_3x5(left_data, right_data, atlas_img, freq_label, output_dir):
    """
    Plot 3x5 grid showing difference maps (left - right) for 3 views x 5 slices each
    
    Parameters:
    -----------
    left_data : ndarray
        Shape (n_subjects, n_regions) - left hemisphere data
    right_data : ndarray
        Shape (n_subjects, n_regions) - right hemisphere data
    atlas_img : nibabel image
        Atlas image
    freq_label : str
        Frequency label (e.g., '5Hz', '20Hz')
    output_dir : str
        Directory to save the figure
    """
    from nilearn import plotting
    
    # Calculate mean difference for each region
    mean_diff = np.mean(left_data - right_data, axis=0)
    
    # Load atlas data
    atlas_data = atlas_img.get_fdata()
    
    # Create difference volume
    diff_volume = np.zeros_like(atlas_data)
    for region_idx in range(len(mean_diff)):
        region_label = region_idx + 1
        diff_volume[atlas_data == region_label] = mean_diff[region_idx]
    
    # Create NIfTI image
    diff_nii = nib.Nifti1Image(diff_volume, atlas_img.affine, atlas_img.header)
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # Axial slices (z-axis) - row 0
    z_coords = np.linspace(-40, 60, 5)
    for i, z in enumerate(z_coords):
        ax = axes[0, i]
        display = plotting.plot_stat_map(
            diff_nii,
            cut_coords=[z],
            display_mode='z',
            axes=ax,
            colorbar=(i == 4),
            cmap='RdBu_r',
            symmetric_cbar=True,
            vmax=np.abs(mean_diff).max(),
            threshold=0.001,
            title=f'Axial z={z:.0f}'
        )
    
    # Sagittal slices (x-axis) - row 1
    x_coords = np.linspace(-50, 50, 5)
    for i, x in enumerate(x_coords):
        ax = axes[1, i]
        display = plotting.plot_stat_map(
            diff_nii,
            cut_coords=[x],
            display_mode='x',
            axes=ax,
            colorbar=(i == 4),
            cmap='RdBu_r',
            symmetric_cbar=True,
            vmax=np.abs(mean_diff).max(),
            threshold=0.001,
            title=f'Sagittal x={x:.0f}'
        )
    
    # Coronal slices (y-axis) - row 2
    y_coords = np.linspace(-70, 30, 5)
    for i, y in enumerate(y_coords):
        ax = axes[2, i]
        display = plotting.plot_stat_map(
            diff_nii,
            cut_coords=[y],
            display_mode='y',
            axes=ax,
            colorbar=(i == 4),
            cmap='RdBu_r',
            symmetric_cbar=True,
            vmax=np.abs(mean_diff).max(),
            threshold=0.001,
            title=f'Coronal y={y:.0f}'
        )
    
    plt.suptitle(f'Difference Map (Left - Right) - {freq_label}\n(Mean correlation difference across subjects)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'difference_map_3x5_{freq_label}.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved 3x5 difference map to {output_path}")

def plot_sta_correlation_4x3(data_dict, atlas_img, correction_type, output_dir):
    """
    Plot 4x3 grid showing STA-to-brain correlations for 4 conditions (3 slices each)
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with keys like 'left_5Hz', 'right_5Hz', 'left_20Hz', 'right_20Hz'
        Each value is a tuple: (mean_correlation, significant_mask, t_stats)
    atlas_img : nibabel image
        Atlas image
    correction_type : str
        'FDR' or 'Bonferroni'
    output_dir : str
        Directory to save the figure
    """
    from nilearn import plotting
    
    atlas_data = atlas_img.get_fdata()
    n_regions = 268
    
    # Define the 4 conditions in order
    conditions = ['left_5Hz', 'right_5Hz', 'left_20Hz', 'right_20Hz']
    condition_labels = ['Left STA - 5Hz', 'Right STA - 5Hz', 'Left STA - 20Hz', 'Right STA - 20Hz']
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    
    for row_idx, (condition, label) in enumerate(zip(conditions, condition_labels)):
        mean_corr, sig_mask, t_stats = data_dict[condition]
        
        # Create volume with mean correlations only for significant regions
        sig_volume = np.zeros_like(atlas_data)
        for region_idx in range(n_regions):
            if sig_mask[region_idx]:
                region_label = region_idx + 1
                sig_volume[atlas_data == region_label] = mean_corr[region_idx]
        
        # Create NIfTI image
        sig_nii = nib.Nifti1Image(sig_volume, atlas_img.affine, atlas_img.header)
        
        # Get vmax for symmetric colorbar
        if np.any(sig_mask):
            vmax = np.abs(mean_corr[sig_mask]).max()
        else:
            vmax = 0.1
        
        # Three slices: axial (z=10), sagittal (x=0), coronal (y=-20)
        slice_coords = [10, 0, -20]
        slice_modes = ['z', 'x', 'y']
        slice_names = ['Axial z=10', 'Sagittal x=0', 'Coronal y=-20']
        
        for col_idx, (coord, mode, slice_name) in enumerate(zip(slice_coords, slice_modes, slice_names)):
            ax = axes[row_idx, col_idx]
            
            n_sig = np.sum(sig_mask)
            title = f'{label}\n{slice_name} (n={n_sig})' if col_idx == 0 else f'{slice_name}'
            
            if n_sig > 0:
                display = plotting.plot_stat_map(
                    sig_nii,
                    cut_coords=[coord],
                    display_mode=mode,
                    axes=ax,
                    colorbar=(col_idx == 2),
                    cmap='RdBu_r',
                    symmetric_cbar=True,
                    vmax=vmax,
                    threshold=0.001,
                    title=title
                )
            else:
                # Plot empty brain if no significant regions
                plotting.plot_stat_map(
                    sig_nii,
                    cut_coords=[coord],
                    display_mode=mode,
                    axes=ax,
                    colorbar=False,
                    title=f'{title}\n(no sig. regions)'
                )
    
    plt.suptitle(f'STA-Brain Envelope Correlations\n({correction_type} corrected, p < 0.05)', 
                 fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'sta_correlation_4x3_{correction_type}.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved 4x3 {correction_type} corrected plot to {output_path}")

def compute_onesample_stats_with_correction(data, alpha=0.05):
    """
    Perform one-sample t-tests for each region and apply corrections
    
    Parameters:
    -----------
    data : ndarray
        Shape (n_subjects, n_regions)
    alpha : float
        Significance level
    
    Returns:
    --------
    mean_corr : array
        Mean correlation for each region
    t_stats : array
        T-statistic for each region
    p_values : array
        Uncorrected p-values
    sig_bonferroni : array
        Boolean mask for Bonferroni-corrected significance
    sig_fdr : array
        Boolean mask for FDR-corrected significance
    """
    n_subjects, n_regions = data.shape
    
    mean_corr = np.mean(data, axis=0)
    t_stats = np.zeros(n_regions)
    p_values = np.zeros(n_regions)
    
    for region_idx in range(n_regions):
        t_stat, p_val = stats.ttest_1samp(data[:, region_idx], 0)
        t_stats[region_idx] = t_stat
        p_values[region_idx] = p_val
    
    # Bonferroni correction
    bonferroni_alpha = alpha / n_regions
    sig_bonferroni = p_values < bonferroni_alpha
    
    # FDR correction (Benjamini-Hochberg)
    try:
        from scipy.stats import false_discovery_control
        fdr_corrected = false_discovery_control(p_values, method='bh')
        sig_fdr = fdr_corrected < alpha
    except ImportError:
        # Manual Benjamini-Hochberg FDR correction
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        n = len(sorted_p_values)
        
        significant_mask = np.zeros(n, dtype=bool)
        for i in range(n-1, -1, -1):
            if sorted_p_values[i] <= ((i + 1) / n) * alpha:
                significant_mask[:i+1] = True
                break
        
        sig_fdr = np.zeros(n, dtype=bool)
        sig_fdr[sorted_indices] = significant_mask
    
    return mean_corr, t_stats, p_values, sig_bonferroni, sig_fdr

# ============================================================================
# GENERATE VISUALIZATIONS FOR EACH FREQUENCY
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Load atlas
atlas_img = nib.load(atlas_path)

# Generate visualizations for 5Hz
if SAVE_RESULTS:
    print("\nGenerating 5Hz visualizations...")
    output_dir_5hz = os.path.join(results_dir, 'cluster_analysis', '5Hz')
    
    # 3x3 cluster plot
    if len(cluster_pvals_5hz) > 0:
        plot_cluster_pvalue_3x3(cluster_img_5hz, atlas_img, '5Hz', output_dir_5hz)
    else:
        print("No significant clusters for 5Hz, skipping 3x3 plot")
    
    # 3x5 difference map
    plot_difference_map_3x5(left_5hz_regions, right_5hz_regions, atlas_img, '5Hz', output_dir_5hz)

# Generate visualizations for 20Hz
if SAVE_RESULTS:
    print("\nGenerating 20Hz visualizations...")
    output_dir_20hz = os.path.join(results_dir, 'cluster_analysis', '20Hz')
    
    # 3x3 cluster plot
    if len(cluster_pvals_20hz) > 0:
        plot_cluster_pvalue_3x3(cluster_img_20hz, atlas_img, '20Hz', output_dir_20hz)
    else:
        print("No significant clusters for 20Hz, skipping 3x3 plot")
    
    # 3x5 difference map
    plot_difference_map_3x5(left_20hz_regions, right_20hz_regions, atlas_img, '20Hz', output_dir_20hz)

# ============================================================================
# GENERATE FDR AND BONFERRONI CORRECTED STA CORRELATION FIGURES
# ============================================================================
if SAVE_RESULTS:
    print("\n" + "="*70)
    print("GENERATING FDR/BONFERRONI CORRECTED STA CORRELATION FIGURES")
    print("="*70)
    
    # Create output directory for corrected figures
    corrected_output_dir = os.path.join(results_dir, 'corrected_sta_correlations')
    os.makedirs(corrected_output_dir, exist_ok=True)
    
    # Compute one-sample statistics for each condition
    print("\nComputing one-sample t-tests for each STA-brain condition...")
    
    # Left 5Hz
    mean_left_5hz, t_left_5hz, p_left_5hz, sig_bonf_left_5hz, sig_fdr_left_5hz = \
        compute_onesample_stats_with_correction(left_5hz_regions)
    print(f"  Left 5Hz: {np.sum(sig_bonf_left_5hz)} Bonferroni sig, {np.sum(sig_fdr_left_5hz)} FDR sig")
    
    # Right 5Hz
    mean_right_5hz, t_right_5hz, p_right_5hz, sig_bonf_right_5hz, sig_fdr_right_5hz = \
        compute_onesample_stats_with_correction(right_5hz_regions)
    print(f"  Right 5Hz: {np.sum(sig_bonf_right_5hz)} Bonferroni sig, {np.sum(sig_fdr_right_5hz)} FDR sig")
    
    # Left 20Hz
    mean_left_20hz, t_left_20hz, p_left_20hz, sig_bonf_left_20hz, sig_fdr_left_20hz = \
        compute_onesample_stats_with_correction(left_20hz_regions)
    print(f"  Left 20Hz: {np.sum(sig_bonf_left_20hz)} Bonferroni sig, {np.sum(sig_fdr_left_20hz)} FDR sig")
    
    # Right 20Hz
    mean_right_20hz, t_right_20hz, p_right_20hz, sig_bonf_right_20hz, sig_fdr_right_20hz = \
        compute_onesample_stats_with_correction(right_20hz_regions)
    print(f"  Right 20Hz: {np.sum(sig_bonf_right_20hz)} Bonferroni sig, {np.sum(sig_fdr_right_20hz)} FDR sig")
    
    # Prepare data dictionaries for plotting
    # FDR corrected
    fdr_data = {
        'left_5Hz': (mean_left_5hz, sig_fdr_left_5hz, t_left_5hz),
        'right_5Hz': (mean_right_5hz, sig_fdr_right_5hz, t_right_5hz),
        'left_20Hz': (mean_left_20hz, sig_fdr_left_20hz, t_left_20hz),
        'right_20Hz': (mean_right_20hz, sig_fdr_right_20hz, t_right_20hz)
    }
    
    # Bonferroni corrected
    bonferroni_data = {
        'left_5Hz': (mean_left_5hz, sig_bonf_left_5hz, t_left_5hz),
        'right_5Hz': (mean_right_5hz, sig_bonf_right_5hz, t_right_5hz),
        'left_20Hz': (mean_left_20hz, sig_bonf_left_20hz, t_left_20hz),
        'right_20Hz': (mean_right_20hz, sig_bonf_right_20hz, t_right_20hz)
    }
    
    # Generate 4x3 figures
    print("\nGenerating 4x3 FDR corrected figure...")
    plot_sta_correlation_4x3(fdr_data, atlas_img, 'FDR', corrected_output_dir)
    
    print("Generating 4x3 Bonferroni corrected figure...")
    plot_sta_correlation_4x3(bonferroni_data, atlas_img, 'Bonferroni', corrected_output_dir)
    
    # Save statistics to CSV
    sta_stats_df = pd.DataFrame({
        'region': np.arange(1, 269),
        'mean_left_5Hz': mean_left_5hz,
        'mean_right_5Hz': mean_right_5hz,
        'mean_left_20Hz': mean_left_20hz,
        'mean_right_20Hz': mean_right_20hz,
        't_left_5Hz': t_left_5hz,
        't_right_5Hz': t_right_5hz,
        't_left_20Hz': t_left_20hz,
        't_right_20Hz': t_right_20hz,
        'p_left_5Hz': p_left_5hz,
        'p_right_5Hz': p_right_5hz,
        'p_left_20Hz': p_left_20hz,
        'p_right_20Hz': p_right_20hz,
        'sig_bonf_left_5Hz': sig_bonf_left_5hz,
        'sig_bonf_right_5Hz': sig_bonf_right_5hz,
        'sig_bonf_left_20Hz': sig_bonf_left_20hz,
        'sig_bonf_right_20Hz': sig_bonf_right_20hz,
        'sig_fdr_left_5Hz': sig_fdr_left_5hz,
        'sig_fdr_right_5Hz': sig_fdr_right_5hz,
        'sig_fdr_left_20Hz': sig_fdr_left_20hz,
        'sig_fdr_right_20Hz': sig_fdr_right_20hz
    })
    sta_stats_path = os.path.join(corrected_output_dir, 'sta_onesample_statistics.csv')
    sta_stats_df.to_csv(sta_stats_path, index=False)
    print(f"\nSaved STA one-sample statistics to {sta_stats_path}")

# ============================================================================
# OLD BRAIN VISUALIZATION CODE (REMOVED)
# ============================================================================
# The following function is kept for reference but not used
def plot_significant_regions_on_brain_OLD(significant_mask, t_stats, freq_label, correction_type):
    """
    Plot significant regions on brain using Shen atlas
    
    Parameters:
    -----------
    significant_mask : array
        Boolean mask of significant regions
    t_stats : array
        T-statistics for each region
    freq_label : str
        Frequency label (e.g., '5Hz', '20Hz')
    correction_type : str
        Type of correction used (e.g., 'FDR', 'Bonferroni')
    """
    # Use the standard Shen atlas path
    atlas_path = '/projectnb/viscog01/yufang/Neural_representaion_proj/data/shen_2mm_268_parcellation.nii.gz'
    
    if not os.path.exists(atlas_path):
        print(f"Atlas not found at {atlas_path}")
        return
    
    # Load the Shen atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    
    # Create a new volume for visualization
    # Initialize with zeros
    sig_volume = np.zeros_like(atlas_data)
    
    # Map significant regions to atlas
    # Shen atlas has labels 1-268 for the 268 regions
    for region_idx in range(n_regions):
        if significant_mask[region_idx]:
            # Region labels in atlas are 1-indexed
            region_label = region_idx + 1
            # Set voxels belonging to this region to the t-statistic value
            sig_volume[atlas_data == region_label] = t_stats[region_idx]
    
    # Create a new NIfTI image
    sig_img = nib.Nifti1Image(sig_volume, atlas_img.affine, atlas_img.header)
    
    # Plot using nilearn if available
    try:
        from nilearn import plotting
        from nilearn import datasets
        
        # Axial view (z-axis)
        plt.figure(figsize=(12, 10))
        display = plotting.plot_stat_map(
            sig_img,
            title=f'Axial View: Significant Regions ({freq_label}, {correction_type})',
            cut_coords=5,
            display_mode='z',
            colorbar=True,
            cmap='RdBu_r',
            symmetric_cbar=True,
            threshold=0.01
        )
        
        if SAVE_RESULTS:
            brain_plot_path = os.path.join(results_dir, f'significant_regions_axial_{freq_label}_{correction_type}.png')
            plt.savefig(brain_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved axial view to {brain_plot_path}")
        else:
            plt.show()
        
        # Sagittal view (x-axis)
        plt.figure(figsize=(12, 10))
        display = plotting.plot_stat_map(
            sig_img,
            title=f'Sagittal View: Significant Regions ({freq_label}, {correction_type})',
            cut_coords=5,
            display_mode='x',
            colorbar=True,
            cmap='RdBu_r',
            symmetric_cbar=True,
            threshold=0.01
        )
        
        if SAVE_RESULTS:
            brain_plot_path = os.path.join(results_dir, f'significant_regions_sagittal_{freq_label}_{correction_type}.png')
            plt.savefig(brain_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved sagittal view to {brain_plot_path}")
        else:
            plt.show()
        
        # Coronal view (y-axis)
        plt.figure(figsize=(12, 10))
        display = plotting.plot_stat_map(
            sig_img,
            title=f'Coronal View: Significant Regions ({freq_label}, {correction_type})',
            cut_coords=5,
            display_mode='y',
            colorbar=True,
            cmap='RdBu_r',
            symmetric_cbar=True,
            threshold=0.01
        )
        
        if SAVE_RESULTS:
            brain_plot_path = os.path.join(results_dir, f'significant_regions_coronal_{freq_label}_{correction_type}.png')
            plt.savefig(brain_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved coronal view to {brain_plot_path}")
        else:
            plt.show()
        
        # Surface plot - use volume to surface projection instead of direct mapping
        try:
            # Get fsaverage surface for plotting
            fsaverage = datasets.fetch_surf_fsaverage()
            
            # Project volume data to surface using nilearn's vol_to_surf
            from nilearn import surface
            
            # Project to left hemisphere surface
            left_surf_values = surface.vol_to_surf(
                sig_img,
                fsaverage.pial_left,
                radius=3.0,
                interpolation='nearest',
                kind='line'
            )
            
            # Project to right hemisphere surface
            right_surf_values = surface.vol_to_surf(
                sig_img,
                fsaverage.pial_right,
                radius=3.0,
                interpolation='nearest',
                kind='line'
            )
            
            # Check if surface values are valid (not all zeros/NaN and have some variation)
            left_has_valid_data = (not np.all(left_surf_values == 0) and 
                                   not np.all(np.isnan(left_surf_values)) and
                                   np.nanmax(left_surf_values) != np.nanmin(left_surf_values))
            
            right_has_valid_data = (not np.all(right_surf_values == 0) and 
                                    not np.all(np.isnan(right_surf_values)) and
                                    np.nanmax(right_surf_values) != np.nanmin(right_surf_values))
            
            # Plot left hemisphere only if we have valid data
            if left_has_valid_data:
                plt.figure(figsize=(12, 10))
                plotting.plot_surf_stat_map(
                    fsaverage.pial_left,
                    left_surf_values,
                    hemi='left',
                    title=f'Left Hemisphere: Significant Regions ({freq_label}, {correction_type})',
                    colorbar=True,
                    cmap='RdBu_r',
                    symmetric_cbar=True,
                    threshold=0.01,
                    bg_map=fsaverage.sulc_left
                )
                
                if SAVE_RESULTS:
                    brain_plot_path = os.path.join(results_dir, f'significant_regions_surface_left_{freq_label}_{correction_type}.png')
                    plt.savefig(brain_plot_path, dpi=300, bbox_inches='tight')
                    print(f"Saved left hemisphere surface view to {brain_plot_path}")
                else:
                    plt.show()
            else:
                print(f"Skipping left hemisphere surface plot: no valid data to display")
            
            # Plot right hemisphere only if we have valid data
            if right_has_valid_data:
                plt.figure(figsize=(12, 10))
                plotting.plot_surf_stat_map(
                    fsaverage.pial_right,
                    right_surf_values,
                    hemi='right',
                    title=f'Right Hemisphere: Significant Regions ({freq_label}, {correction_type})',
                    colorbar=True,
                    cmap='RdBu_r',
                    symmetric_cbar=True,
                    threshold=0.01,
                    bg_map=fsaverage.sulc_right
                )
                
                if SAVE_RESULTS:
                    brain_plot_path = os.path.join(results_dir, f'significant_regions_surface_right_{freq_label}_{correction_type}.png')
                    plt.savefig(brain_plot_path, dpi=300, bbox_inches='tight')
                    print(f"Saved right hemisphere surface view to {brain_plot_path}")
                else:
                    plt.show()
            else:
                print(f"Skipping right hemisphere surface plot: no valid data to display")
                
        except Exception as e:
            print(f"Error creating surface plot: {e}")
            print(f"Skipping surface visualization.")
            
    except ImportError:
        print("nilearn not available. Saving NIfTI file instead.")
        if SAVE_RESULTS:
            nifti_path = os.path.join(results_dir, f'significant_regions_{freq_label}_{correction_type}.nii.gz')
            nib.save(sig_img, nifti_path)
            print(f"Saved significant regions as NIfTI to {nifti_path}")

# Old plotting calls removed - now using cluster-based visualizations above

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nNumber of subjects analyzed: {len(df)}")
print(f"Number of excluded subjects: {len(excluded_subjects)}")
print(f"Number of regions in Shen atlas: {n_regions}")
print(f"\nComparison: Left vs Right hemisphere at 5Hz and 20Hz")
print(f"\nAnalysis 1A (Average across all regions - 5Hz):")
print(f"  Left 5Hz mean: {np.mean(left_5hz_avg):.4f}")
print(f"  Right 5Hz mean: {np.mean(right_5hz_avg):.4f}")
print(f"  Paired t-test p-value: {p_val_5hz:.6f}")
print(f"  Significant: {'Yes' if p_val_5hz < 0.05 else 'No'}")
print(f"\nAnalysis 1B (Average across all regions - 20Hz):")
print(f"  Left 20Hz mean: {np.mean(left_20hz_avg):.4f}")
print(f"  Right 20Hz mean: {np.mean(right_20hz_avg):.4f}")
print(f"  Paired t-test p-value: {p_val_20hz:.6f}")
print(f"  Significant: {'Yes' if p_val_20hz < 0.05 else 'No'}")
print(f"\nAnalysis 2A (Region-by-region - 5Hz):")
print(f"  Significant regions (uncorrected): {np.sum(region_p_values_5hz < 0.05)}/{n_regions}")
print(f"  Significant regions (Bonferroni): {np.sum(significant_regions_bonferroni_5hz)}/{n_regions}")
print(f"  Significant regions (FDR): {np.sum(significant_regions_fdr_5hz)}/{n_regions}")
print(f"\nAnalysis 2B (Region-by-region - 20Hz):")
print(f"  Significant regions (uncorrected): {np.sum(region_p_values_20hz < 0.05)}/{n_regions}")
print(f"  Significant regions (Bonferroni): {np.sum(significant_regions_bonferroni_20hz)}/{n_regions}")
print(f"  Significant regions (FDR): {np.sum(significant_regions_fdr_20hz)}/{n_regions}")
print(f"\nAnalysis 3A (Cluster Permutation - 5Hz):")
if len(cluster_pvals_5hz) > 0:
    print(f"  Number of clusters: {len(cluster_pvals_5hz)}")
    print(f"  Significant clusters (p < 0.05): {np.sum(cluster_pvals_5hz < 0.05)}")
else:
    print(f"  No clusters found")
print(f"\nAnalysis 3B (Cluster Permutation - 20Hz):")
if len(cluster_pvals_20hz) > 0:
    print(f"  Number of clusters: {len(cluster_pvals_20hz)}")
    print(f"  Significant clusters (p < 0.05): {np.sum(cluster_pvals_20hz < 0.05)}")
else:
    print(f"  No clusters found")

if SAVE_RESULTS:
    # Save summary report
    summary_path = os.path.join(results_dir, 'statistics', 'analysis_summary_left_right_comparison.txt')
    with open(summary_path, 'w') as f:
        f.write("STA-Brain Envelope Correlation Analysis\n")
        f.write("Left vs Right Hemisphere Comparison (5Hz and 20Hz)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of subjects: {len(df)}\n")
        f.write(f"Number of excluded subjects: {len(excluded_subjects)}\n")
        f.write(f"Excluded subjects: {excluded_subjects}\n")
        f.write(f"Number of Shen atlas regions: {n_regions}\n")
        f.write(f"Number of permutations: {N_PERMUTATIONS}\n\n")
        
        f.write("ANALYSIS 1A: Average Correlation Across All Regions - 5Hz\n")
        f.write("-" * 50 + "\n")
        f.write(f"Left 5Hz:   Mean={np.mean(left_5hz_avg):.4f} ± {np.std(left_5hz_avg):.4f}\n")
        f.write(f"Right 5Hz:  Mean={np.mean(right_5hz_avg):.4f} ± {np.std(right_5hz_avg):.4f}\n")
        f.write(f"Paired t-test: t={t_stat_5hz:.4f}, p={p_val_5hz:.6f}\n")
        f.write(f"Cohen's d: {cohens_d_5hz:.4f}\n\n")
        
        f.write("ANALYSIS 1B: Average Correlation Across All Regions - 20Hz\n")
        f.write("-" * 50 + "\n")
        f.write(f"Left 20Hz:  Mean={np.mean(left_20hz_avg):.4f} ± {np.std(left_20hz_avg):.4f}\n")
        f.write(f"Right 20Hz: Mean={np.mean(right_20hz_avg):.4f} ± {np.std(right_20hz_avg):.4f}\n")
        f.write(f"Paired t-test: t={t_stat_20hz:.4f}, p={p_val_20hz:.6f}\n")
        f.write(f"Cohen's d: {cohens_d_20hz:.4f}\n\n")
        
        f.write("ANALYSIS 2A: Region-by-Region Comparison - 5Hz\n")
        f.write("-" * 50 + "\n")
        f.write(f"Bonferroni corrected alpha: {bonferroni_alpha:.6f}\n")
        f.write(f"Significant regions (uncorrected): {np.sum(region_p_values_5hz < 0.05)}/{n_regions}\n")
        f.write(f"Significant regions (Bonferroni): {np.sum(significant_regions_bonferroni_5hz)}/{n_regions}\n")
        f.write(f"Significant regions (FDR): {np.sum(significant_regions_fdr_5hz)}/{n_regions}\n\n")
        
        f.write("ANALYSIS 2B: Region-by-Region Comparison - 20Hz\n")
        f.write("-" * 50 + "\n")
        f.write(f"Bonferroni corrected alpha: {bonferroni_alpha:.6f}\n")
        f.write(f"Significant regions (uncorrected): {np.sum(region_p_values_20hz < 0.05)}/{n_regions}\n")
        f.write(f"Significant regions (Bonferroni): {np.sum(significant_regions_bonferroni_20hz)}/{n_regions}\n")
        f.write(f"Significant regions (FDR): {np.sum(significant_regions_fdr_20hz)}/{n_regions}\n\n")
        
        f.write("ANALYSIS 3A: Cluster Permutation Test - 5Hz\n")
        f.write("-" * 50 + "\n")
        if len(cluster_pvals_5hz) > 0:
            f.write(f"Number of clusters: {len(cluster_pvals_5hz)}\n")
            f.write(f"Significant clusters (p < 0.05): {np.sum(cluster_pvals_5hz < 0.05)}\n")
            for i, (size, pval) in enumerate(zip(cluster_sizes_5hz, cluster_pvals_5hz)):
                f.write(f"  Cluster {i+1}: size={size} voxels, p={pval:.4f}\n")
        else:
            f.write("No clusters found\n")
        f.write("\n")
        
        f.write("ANALYSIS 3B: Cluster Permutation Test - 20Hz\n")
        f.write("-" * 50 + "\n")
        if len(cluster_pvals_20hz) > 0:
            f.write(f"Number of clusters: {len(cluster_pvals_20hz)}\n")
            f.write(f"Significant clusters (p < 0.05): {np.sum(cluster_pvals_20hz < 0.05)}\n")
            for i, (size, pval) in enumerate(zip(cluster_sizes_20hz, cluster_pvals_20hz)):
                f.write(f"  Cluster {i+1}: size={size} voxels, p={pval:.4f}\n")
        else:
            f.write("No clusters found\n")
    
    print(f"\nSaved summary to {summary_path}")
    print(f"\nAll results saved to: {results_dir}")
    print(f"  - Statistics: {os.path.join(results_dir, 'statistics')}")
    print(f"  - Boxplots: {os.path.join(results_dir, 'boxplots')}")
    print(f"  - Cluster analysis (5Hz): {os.path.join(results_dir, 'cluster_analysis', '5Hz')}")
    print(f"  - Cluster analysis (20Hz): {os.path.join(results_dir, 'cluster_analysis', '20Hz')}")
    print(f"  - Corrected STA correlations: {os.path.join(results_dir, 'corrected_sta_correlations')}")
else:
    print(f"\nSaving is disabled. Set SAVE_RESULTS=True to save results to disk.")
