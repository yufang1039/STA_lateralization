import numpy as np
import multiprocessing as mp
import subprocess
import os
from settings import Settings


def non_linear_registration(case, freq, sensor):
    '''

    Usage:

    antsRegistrationSyN.sh -d ImageDimension -f FixedImage -m MovingImage -o OutputPrefix

     -d:  ImageDimension: 2 or 3 (for 2 or 3 dimensional registration of single volume)

     -f:  Fixed image(s) or source image(s) or reference image(s)

     -m:  Moving image(s) or target image(s)

     -o:  OutputPrefix: A prefix that is prepended to all output files.

     -n:  Number of threads

    '''
    
    subdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
    freesurfer_home = os.environ.get('FREESURFER_HOME')
    if not freesurfer_home:
        raise ValueError("FREESURFER_HOME environment variable must be set.")
    fsaverage = f'{freesurfer_home}/subjects/fsaverage/mri'
    input_file = f'{subdir}/{case}/mne_files/{case}_true_7.8_{freq}_{sensor}_corr.nii.gz'
    output_file = f'{subdir}/{case}/mne_files/{case}_ants'
    bash_cmd = f'antsRegistrationSyNQuick.sh -d 3 -f {fsaverage}/brain.mgz -m {input_file} -o {output_file} -n 4'
    print(bash_cmd)
    subprocess.check_output(bash_cmd, shell=True)


if __name__ == '__main__':

    settings = Settings()

    data_params = settings['DATA']
    hyper_params = settings['PARAMS']
    common_params = settings['COMMON']

    cases = data_params['cases']
    subjects_dir = common_params['subjects_dir']
    spacing = hyper_params['vol_spacing']
    freqs = hyper_params['freqs']
    flag = hyper_params['ortho_flag']
    sensor = hyper_params['sensor']
    njobs = hyper_params['njobs']

    
    with open(cases) as f:
        case_list = f.read().splitlines()
    for freq in freqs:
        for label in sensor:
            pool = mp.Pool(processes=njobs)
            for index, subject in enumerate(case_list):
                pool.apply_async(non_linear_registration, args=[subject, freq, label])
            pool.close()
            pool.join()
