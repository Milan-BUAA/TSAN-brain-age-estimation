# Data Preprocessing Pipeline

## Data preprocessing script

As describe in our paper, all MRIs in dataset were processed by using a standard preprocessing pipeline with FSL 5.10, including brain extraction and nonlinear registration to the standard MNI space. All MRIs after preprocessing have voxel size of 91 $\times$ 109 $\times$ 91 with isotropic spatial resolution of $2 mm^{3}$.

Specifically, using the original  [fsl_anat](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/fsl_anat)  preprocessing script in FSL could not directly get the nonlinear reregistration image with brain extraction, so we made a simple modification to [fsl_anat](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/fsl_anat) script and name it "fsl_anat_tsan".

(See #REGISTRATION AND BRAIN EXTRACTION in the [fsl_anat_tsan](https://github.com/Milan-BUAA/TSAN-brain-age-estimation/blob/master/data_preprocessing/fsl_anat_tsan) file for details of the modification. )

To use [fsl_anat_tsan](https://github.com/Milan-BUAA/TSAN-brain-age-estimation/blob/master/data_preprocessing/fsl_anat_tsan) preprocessing script, you need to put the file in FSL command folder (e.g /usr/lib/fsl/5.1) and then you can execute it like any other FSL command.

The main products of this preprocessing script include:

- *The image after linear registration*     (T1_to_MNI_lin.nii.gz)
- *The image after non-linear registration*  (T1_to_MNI_nonlin.nii.gz)
- ***The image after brain extraction and non-linear registration*** (**T1_to_MNI_nonlin_brain.nii.gz**)

The image after brain extraction and non-linear registration is what we need in this paper.

You can use simple *For Loop* command to execute the script:

```shell
for file in ./raw_data/*; do fsl_anat_tsan -i $file; done
```

## Multi-threading Data preprocessing script

Due to the large number of images that need to be preprocessed, we also wrote a simple script that can execute fsl_anat_tsan in multiple threads,  named [multi-thread-prep.sh](https://github.com/Milan-BUAA/TSAN-brain-age-estimation/blob/master/data_preprocessing/multi-thread-prep.sh) 

In *multi-thread-prep.sh* file, you only need to set raw data folder, and repeatedly set the start index and number of image need to be processed.  

```
bash multi-thread-prep.sh
```



------

In summary, after executing our preprocessing pipeline,  the data organization will change

**From**

```
Raw data Folder-----
          sub-0001.nii.gz
          sub-0002.nii.gz
          .......
```

**To**

```
Processed data Folder-----
          sub-0001/
                sub-001.nii.gz 
                sub-001.anat/
                    T1_to_MNI_lin.nii.gz
                    T1_to_MNI_nonlin.nii.gz
                    T1_to_MNI_nonlin_brain.nii.gz
                    .......
          sub-0002/
                sub-001.nii.gz
                sub-001.anat/
                    T1_to_MNI_lin.nii.gz
                    T1_to_MNI_nonlin.nii.gz
                    T1_to_MNI_nonlin_brain.nii.gz
                    .......
          .......
```

## Reorganize the preprocessed resulting images

According to the above image preprocessing scripts, we will get a variety of preprocessing results of the original image. However, as for brain age estimation with TSAN, our models were trained by image with brain extraction and non-linear registration. Therefore, we wrote a simple script to reorganize the preprocessed image files and classify them for subsequent training and testing

```shell
python Extract-File-From-Prep.py --sdir /data/brain_age_estimation/ --tdir /data/brain_age_estimation_prep-org/ --keyword .anat
```

After excuting this reorgnization script, the data organization will change

**From**

```
Processed data Folder-----
          sub-0001/
                sub-001.nii.gz 
                sub-001.anat/
                    T1_to_MNI_lin.nii.gz
                    T1_to_MNI_nonlin.nii.gz
                    T1_to_MNI_nonlin_brain.nii.gz
                    .......
          sub-0002/
                sub-001.nii.gz
                sub-001.anat/
                    T1_to_MNI_lin.nii.gz
                    T1_to_MNI_nonlin.nii.gz
                    T1_to_MNI_nonlin_brain.nii.gz
                    .......
          .......
```

**To**

```
Reorganize data Folder-----
        linear/
            sub-001_lin.nii.gz
            sub-002_lin.nii.gz
            sub-003_lin.nii.gz
            .......

        nonlinear/ 
            sub-001_nonlin.nii.gz
            sub-002_nonlin.nii.gz
            sub-003_nonlin.nii.gz      
            .......

        # The images after brain extraction and non-linear registration are ready for training or evaluating the TSAN model
        nonlin_brain/
            sub-001_nonlin_brain.nii.gz
            sub-002_nonlin_brain.nii.gz
            sub-003_nonlin_brain.nii.gz
            .......
```

## Train-Test split for preprocessed images

According the above steps, the preprocessed T1-weighted images have now been obtained and reorganized in the same folder.

If you want to do transfer learning or fine tuning, you can excute our *train-test-split.py* code to divide the preprocessed data into training, validation and test set. In our paper, Train: Val: Test = 70% : 15% : 15%

```
python train-test-split.py --sdir /data/brain_age_estimation_prep-org/nonlin_brain --tdir /data/brain_age_estimation_transfer_learning/ --train 0.75 --val 0.15 --test 0.15
```

After Train-Test-split, the data organization will change to

```
Train Folder-----
          sub-0001_nonlin_brai.nii.gz
          sub-0002_nonlin_brai.nii.gz
          .......

Validation Folder-----
          sub-0003_nonlin_brai.nii.gz
          sub-0004_nonlin_brai.nii.gz
          .......
Test Folder-----
          sub-0005_nonlin_brai.nii.gz
          sub-0006_nonlin_brai.nii.gz
          .......
          
.......
```
