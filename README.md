# Brain Age Estimation From MRI Using Cascade Networks with Ranking Loss

Pytorch codes for the paper "[Brain Age Estimation From MRI Using Cascade Networks with Ranking Loss](https://doi.org/10.1109/TMI.2021.3085948)", *IEEE transactions on Medical Imaging*, 2021 

## Abstract

In this paper, a novel 3D convolutional network, called as two-stage-age-network (TSAN), is proposed to estimate brain age from T1-weighted MRI data. Compared with existing methods, TSAN has the following improvements: 

- TSAN uses a two-stage cascade network architecture, where the first-stage network estimates a rough brain age, then the second-stage network estimates the brain age more accurately from the discretized brain age by the first stage network. 
- To our knowledge, TSAN is the first work to apply novel ranking losses in brain age estimation, together with the traditional mean square error (MSE) loss. 
- Third, densely connected paths are used to combine feature maps with different scales. 

![TSAN](/imgs/TSAN.png)

## Using the code:

- ### **Clone this repository:**

```
git clone https://github.com/Milan-BUAA/TSAN-brain-age-estimation.git
cd TSAN-brain-age-estimation
```


- ### **To install all the dependencies using pip:**
The code is stable using Python 3.8, to use it you will need:
 * Python >= 3.8
 * Pytorch >= 1.7
 * numpy
 * nibabel
 * tensorboardX
 * sklearn
 * pandas

Install dependencies with

```
pip install -r requirements.txt
```

- ### **Data Pre-Processing:**
  Before performing training and testing, all MRI data needs to be preprocessed. All MRIs in datasets were processed by using a standardpreprocessing pipeline with FSL, including nonlinear registration to the standard MNI space and brain extraction. All MRIs after preprocessing have voxel size of $91 \times 109 \times 91$ with isotropic spatial resolution of $2 mm^{3}$. Details data preprocessing method see [here](https://github.com/Milan-BUAA/TSAN-brain-age-estimation/tree/master/data_preprocessing).

- ### **Training Command:**

Change the model_name, data_path and other settings to train them

```
# For training the frist stage brain age estimation network
bash script/bash_train_first_stage.sh
```

```
# For training the second stage brain age estimation network
# with the fisrt stage network pretrained model
bash script/bash_train_second_stage.sh
```

- ### **Testing Command:**

Change the model_name, data_path and other settings to inference them



```
# For testing the frist stage brain age estimation network
bash script/bash_test_first_stage.sh
```

```
# For testing the second stage brain age estimation network with 
# the first stage network
bash script/bash_test_second_stage.sh
```

## Pre-trained Model
Download the pretrained first-stage ScaleDense model and the second-stage model: [Beihang Cloud](https://bhpan.buaa.edu.cn:443/link/7C6286240B710575452B3E8220032732)

## Datasets

Please check related websites for getting the datasets used in this paper:

[ADNI](http://adni.loni.usc.edu/)

[OASIS](https://www.oasis-brains.org/)

[PAC 2019 website archive](https://web.archive.org/web/20200214101600/https://www.photon-ai.com/pac2019)

### Data Structure

Prepare the dataset in the following format for easy use of the code.  

- Train, validation and test should contain completely unduplicated T1-weighted image samples.
- The Excel file should include image file names, chronological age and sex labels ('0' for female and '1' for male) for all samples from the three datasets.

```
Train Folder-----
          sub-0001.nii.gz
          sub-0002.nii.gz
          .......

Validation Folder-----
          sub-0003.nii.gz
          sub-0004.nii.gz
          .......
Test Folder-----
          sub-0005.nii.gz
          sub-0006.nii.gz
          .......
          
Dataset.xls 

sub-0001.nii.gz     60     1
sub-0002.nii.gz     74     0
.......
```

## Reference

If this repository is useful for your work, please cite the references:

[1] Jian Cheng, Ziyang Liu, Hao Guan, Zhenzhou Wu, Haogang Zhu, Jiyang Jiang, Wei Wen, Dacheng Tao, Tao Liu, "[Brain Age Estimation From MRI Using Cascade Networks with Ranking Loss](https://doi.org/10.1109/TMI.2021.3085948)." *IEEE transactions on Medical Imaging*, 2021. [[arxiv](https://arxiv.org/pdf/2106.03052.pdf)]

[2] Ziyang Liu, Jian Cheng, Haogang Zhu, Jicong Zhang, and Tao Liu, "[Brain Age Estimation from MRI Using a Two-Stage Cascade Network with Ranking Loss](https://doi.org/10.1007/978-3-030-59728-3_20)." In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, pp. 198-207. Springer, Cham, 2020.



