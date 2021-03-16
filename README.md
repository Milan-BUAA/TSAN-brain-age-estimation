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
git clone https://github.com/liuziyang1106/TSAN
cd TSAN
```

The code is stable using Python 3.8, Pytorch 1.7

- ### **To install all the dependencies using pip:**	

```
pip install -r requirments.txt
```

- ### **Training Command:**

```
# For training frist stage brian age estimation network
bash bash_train_first_stage.sh
```

```
# For training second stage brian age estimation network
bash bash_train_second_stage.sh
```

​		Change model_name, data_path and other settings to train them

- ### **Testing Command:**

```
# For Test frist stage brian age estimation network
bash bash_test_first_stage.sh
```

```
# For Tesing second stage brian age estimation network
bash bash_test_second_stage.sh
```

## Datasets

Please check related websites for getting the datasets used in this paper:

[ADNI](http://adni.loni.usc.edu/)

[OASIS](https://www.oasis-brains.org/)

[PAC 2019 website archive](https://web.archive.org/web/20200214101600/https://www.photon-ai.com/pac2019)

## To cite

**If this repository was useful to your work, please consider citing us:**

Liu, Ziyang, et al. "Brain Age Estimation from MRI Using a Two-Stage Cascade Network with Ranking Loss." *International Conference on Medical Image Computing and Computer-Assisted Intervention*. Springer, Cham, 2020.[https://doi.org/10.1007/978-3-030-59728-3_20]

```
@inproceedings{liu2020brain,
  title={Brain Age Estimation from MRI Using a Two-Stage Cascade Network with Ranking Loss},
  author={Liu, Ziyang and Cheng, Jian and Zhu, Haogang and Zhang, Jicong and Liu, Tao},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={198--207},
  year={2020},
  organization={Springer}
}
```

**Please consider star and/or watch this repository if you find it helpful, as we will keep updating this repository for pre-trained models and weights.**

