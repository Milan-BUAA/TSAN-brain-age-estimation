# Brain Age Estimation From MRI Using Cascade Networks with Ranking Loss

Pytorch Code for the paper "Brain Age Estimation From MRI Using Cascade Networks with Ranking Loss" 

## Abstract

In this paper, a novel 3D convolutional network, called as two-stage-age-network (TSAN), is proposed to estimate brain age from T1-weighted MRI data. Compared with existing methods, TSAN has the following improvements: 

- TSAN uses a two-stage cascade network architecture, where the first-stage network estimates a rough brain age, then the second-stage network estimates the brain age more accurately from the discretized brain age by the first stage network. 
- To our knowledge, TSAN is the first work to apply novel ranking losses in brain age estimation, together with the traditional mean square error (MSE) loss. 
- Third, densely connected paths are used to combine feature maps with different scales. 

## Using the code:

- ### **Clone this repository:**

```
git clone https://github.com/liuziyang1106/TSAN
cd TSAN
```

The code is stable using Python 3.6, Pytorch 1.4.0

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

[PAC 2019]()