import os
import shutil
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr,spearmanr
import pandas as pd

# ==========-   load  prediction and target  ========== #

def bias_correction(target, prediction):
    age_gap = prediction-target
    print('number of sample: ',age_gap.shape[0])

    # ==========   reshape numpy array   ========== #
    prediction = np.expand_dims(prediction,axis=1)
    target = np.expand_dims(target,axis=1)

    reg = LinearRegression()
    reg.fit(target,prediction-target)
    print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

    # ==========   bias correction   ========== #
    correct_prediction = prediction - (reg.coef_[0][0]*target + reg.intercept_[0])
    correct_age_gap = correct_prediction - target
    correct_age_gap = np.squeeze(correct_age_gap,axis=1)

    # ==========   reshape the corrected brain age numpy array and save it   ========== #
    target = np.squeeze(target,axis=1)
    prediction = np.squeeze(prediction,axis=1)
    correct_prediction = np.squeeze(correct_prediction,axis=1)

    # ==========  compute the MAE between predicted age and true age   ========== #

    MAE = mean_absolute_error(target,prediction)
    MAE_bc = mean_absolute_error(target,correct_prediction)

    PCC_age = np.corrcoef(target,prediction)[0][1]
    PCC_age_bc = np.corrcoef(target,correct_prediction)[0][1]

    SRCC_age = spearmanr(target,prediction,axis=1)[0]
    SRCC_age_bc = spearmanr(target,correct_prediction,axis=1)[0]


    PCC_gap = np.corrcoef(target,age_gap)[0][1]
    PCC_gap_bc = np.corrcoef(target,correct_age_gap)[0][1]

    SRCC_gap = spearmanr(target,age_gap,axis=1)[0]
    SRCC_gap_bc = spearmanr(target,correct_age_gap,axis=1)[0]
    # ==========  compute the pearson's corrlation coefication between age gap and true age   ========== #
    # print('=========================')
    # print('Before bias correction CC',)
    # print('After bias correction CC',)

    # print('Before MAE',)
    # print('After correction MAE',)

    # # ==========  compute the spearman's rank corrlation coefication between age gap and true age   ========== #
    # print('=========================')
    # print('Before correction sprearman',)
    # print('After correction sprearman ',)
    result = [MAE, PCC_age, SRCC_age, PCC_gap,SRCC_gap,
              MAE_bc, PCC_age_bc, SRCC_age_bc, PCC_gap_bc, SRCC_gap_bc]
    
    return result

if __name__ == "__main__":
    root = './TMI_result/supply_result/TMI_sub_result/deep-wide'
    model_list = []


    result_summary = np.zeros((len(os.listdir(root)),10))

    for file_idx in range(len(os.listdir(root))):
        target = np.load(os.path.join(root,os.listdir(root)[file_idx]))['target']
        prediction = np.load(os.path.join(root,os.listdir(root)[file_idx]))['prediction']
        model = os.listdir(root)[file_idx].split('.')[0]
        result = bias_correction(target, prediction)
        model_list.append(model)
        for idx in range(10):
            result_summary[file_idx, idx] = result[idx]
    model_list = np.expand_dims(np.asarray(model_list),axis=1) 
    print(model_list.shape)   
    print(result_summary.shape)
    result = np.concatenate((model_list, result_summary),axis=1)
    print(result)
    result = pd.DataFrame(result)
    result.to_excel('./TMI_result/supply_result/deep-wide_result_summary.xlsx')
        

