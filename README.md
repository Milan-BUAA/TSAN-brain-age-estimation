# Brain Age Estimation From MRI Using Cascade Networks with Ranking Loss

Pytorch Code for the paper "Brain Age Estimation From MRI Using Cascade Networks with Ranking Loss" 

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
cd TSAN
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

- ### **Training Command:**

Change the model_name, data_path and other settings to train them

```
# For training the frist stage brain age estimation network
bash bash_train_first_stage.sh
```

```
# For training the second stage brain age estimation network
# with the fisrt stage network pretrained model
bash bash_train_second_stage.sh
```

- ### **Testing Command:**

Change the model_name, data_path and other settings to inference them



```
# For testing the frist stage brain age estimation network
bash bash_test_first_stage.sh
```

```
# For testing the second stage brain age estimation network with 
# the first stage network
bash bash_test_second_stage.sh
```

## Pre-trained Model
Download the ScaleDense and The second stage ScaleDense pretrained model: [Beihang Cloud](https://bhpan.buaa.edu.cn:443/link/7C6286240B710575452B3E8220032732)

## Datasets

Please check related websites for getting the datasets used in this paper:

[ADNI](http://adni.loni.usc.edu/)

[OASIS](https://www.oasis-brains.org/)

[PAC 2019 website archive](https://web.archive.org/web/20200214101600/https://www.photon-ai.com/pac2019)

**Please consider star and/or watch this repository if you find it helpful, as we will keep updating this repository for pre-trained models and weights.**

     