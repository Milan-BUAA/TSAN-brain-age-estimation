# Brain Age Estimation From MRI Using Cascade Networks with Ranking Loss

Pytorch Code for the paper "Brain Age Estimation From MRI Using Cascade Networks with Ranking Loss" 

## Abstract

In this paper, a novel 3D convolutional network, called as two-stage-age-network (TSAN), is proposed to estimate brain age from T1-weighted MRI data. Compared with existing methods, TSAN has the following improvements: 

- TSAN uses a two-stage cascade network architecture, where the first-stage network estimates a rough brain age, then the second-stage network estimates the brain age more accurately from the discretized brain age by the first stage network. 
- To our knowledge, TSAN is the first work to apply novel ranking losses in brain age estimation, together with the traditional mean square error (MSE) loss. 
- Third, densely connected paths are used to combine feature maps with different scales. 

The experiments with 6586 MRIs showed that TSAN could provide accurate brain age estimation, yielding mean absolute error (MAE) of 2.428 and Pearson’s correlation coefficient (PCC) of 0.985, between the estimated and chronological ages. Furthermore, using the brain age gap between brain age and chronological age as a biomarker, Alzheimer’s disease (AD) and Mild Cognitive Impairment (MCI) can be distinguished from healthy control (HC) subjects by support vector machine (SVM). Classification AUC in AD/HC and MCI/HC was 0.931 and 0.899, respectively. It showed that brain age gap is an effective biomarker associated with risk of dementia, and has potential for early-stage dementia risk screening.