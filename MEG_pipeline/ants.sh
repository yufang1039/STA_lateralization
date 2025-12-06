#!/bin/bash

filename='/home/senthilp/caesar/camcan/cc700/freesurfer_output/full.txt'

data_dir='/home/senthilp/caesar/camcan/cc700/freesurfer_output/register_volume'
freesurfer_dir='/home/senthilp/caesar/camcan/cc700/freesurfer_output'

while read -r case
do

templateFile="shen_1mm_268_parcellation.mgz"
output_name="shen_freesurfer.mgz"

nii_name="shen_freesurfer.nii.gz"
#templateFile="power264MNI.mgh"
#output_name="power_freesurfer.mgz"

#echo antsRegistrationSyNQuick.sh -d 3 -f ${freesurfer_dir}/${case}/mri/brain.mgz -m ${data_dir}/shen_greyscale.nii.gz -o ${freesurfer_dir}/${case}/mri/test -n 4

echo antsApplyTransforms -d 3 -i ${data_dir}/${templateFile} -r ${freesurfer_dir}/${case}/mri/aparc.a2009s+aseg.mgz -o ${freesurfer_dir}/${case}/mri/${output_name} -t ${freesurfer_dir}/${case}/mri/test1Warp.nii.gz -t ${freesurfer_dir}/${case}/mri/test0GenericAffine.mat --interpolation NearestNeighbor

if [ ! -f "${freesurfer_dir}/${case}/mri/${output_name}" ]; then
	echo "converting nii to mgz"
	mri_convert ${freesurfer_dir}/${case}/mri/${nii_name} ${freesurfer_dir}/${case}/mri/${output_name}
fi

#echo antsApplyTransforms -d 3 -i ${data_dir}/shen_1mm_268_parcellation.nii.gz -r ${freesurfer_dir}/${case}/mri/aparc.a2009s+aseg.mgz -o ${freesurfer_dir}/${case}/mri/shen_freesurfer.nii.gz -t ${freesurfer_dir}/${case}/mri/test1Warp.nii.gz -t ${freesurfer_dir}/${case}/mri/test0GenericAffine.mat --interpolation NearestNeighbor

#echo antsApplyTransforms -d 3 -i ${data_dir}/shen_1mm_268_parcellation.nii.gz -r ${freesurfer_dir}/${case}/mri/aparc.a2009s+aseg.mgz -o ${freesurfer_dir}/${case}/mri/shen_freesurfer.mgz -t ${freesurfer_dir}/${case}/mri/test1Warp.nii.gz -t ${freesurfer_dir}/${case}/mri/test0GenericAffine.mat --interpolation NearestNeighbor


#echo antsApplyTransforms -d 3 -i ${data_dir}/power264MNI.mgh -r ${freesurfer_dir}/${case}/mri/aparc.a2009s+aseg.mgz -o ${freesurfer_dir}/${case}/mri/power_freesurfer.mgz -t ${freesurfer_dir}/${case}/mri/test1Warp.nii.gz -t ${freesurfer_dir}/${case}/mri/test0GenericAffine.mat --interpolation NearestNeighbor

done < ${filename}
