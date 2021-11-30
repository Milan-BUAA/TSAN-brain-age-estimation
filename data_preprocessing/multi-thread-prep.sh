#!/bin/bash
# this script is to perform preprocess of T1 images:
# include linear reg, non linear reg and brain extraction

## directory
ROOTDIR=/data/brain_age_estimation/
cd ${ROOTDIR}
files=(ls *.nii.gz)
pip (){
    local sta=$1
    local num=$2
    echo "process $sta : $((sta+num))"
    for file in ${files[@]:$sta:$num};do

    # echo $file
    
    fsl_anat_tsan -i $file
    done
}
    
pip 1 300 &
pip 301 300&
pip 601 300 &
pip 901 300 &
pip 1201 300 &
pip 1501 300 &
pip 1801 300 &
pip 2101 300 &
pip 2401 300 &
pip 2701 300 &
