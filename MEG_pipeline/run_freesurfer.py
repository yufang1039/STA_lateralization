import os
import subprocess
from settings import Settings
import multiprocessing as mp
import time
from functools import wraps

os.putenv("SUBJECTS_DIR", "/home/senthilp/caesar/camcan/cc700/freesurfer_output/new")
os.putenv("FREESURFER_HOME", "/home/senthilp/freesurfer")
os.system("echo $SUBJECTS_DIR")
os.system("echo $FREESURFER_HOME")


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print (f'@timefn: {fn.__name__} took {convert(t2-t1)} (hh:mm:ss)')
        return result
    return measure_time


@timefn
def freesurfer_recon_all(T1_mri, subject):
    bash_cmd = f'recon-all -s {subject} -i {T1_mri} -all -threads 12'
    print(bash_cmd)
    subprocess.check_output(bash_cmd, shell=True)


if __name__ == '__main__':

    settings = Settings()
    data_params = settings['DATA']
    hyper_params = settings['PARAMS']
    common_params = settings['COMMON']
    cases = data_params['cases']
    njobs = hyper_params['njobs']
    anat_dir = data_params['anat_dir']

    with open(cases) as f:
        case_list = f.read().splitlines()

    pool = mp.Pool(processes=njobs)
    for subject in case_list:
        T1_mri = f'{anat_dir}/{subject}/anat/{subject}_T1w.nii.gz'
        pool.apply_async(freesurfer_recon_all, args=[T1_mri, subject])
    pool.close()
    pool.join()
