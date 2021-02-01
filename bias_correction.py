import os

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr,spearmanr

# ==========-   load  prediction and target  ========== #
root = './train_result/'
target = np.load(os.path.join(root,'target.npy'))
prediction = np.load(os.path.join(root,'prediction.npy'))
predicted_age_difference = prediction-target
print('number of sample: ',predicted_age_difference.shape[0])

# ==========   reshape numpy array   ========== #
prediction = np.expand_dims(prediction,axis=1)
target = np.expand_dims(target,axis=1)

reg = LinearRegression()
reg.fit(target,prediction-target)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

# ==========   bias correction   ========== #
correct_prediction = prediction - (reg.coef_[0][0]*target + reg.intercept_[0])
correct_gap = correct_prediction - target
correct_gap = np.squeeze(correct_gap,axis=1)

# ==========   reshape the corrected brain age numpy array and save it   ========== #
target = np.squeeze(target,axis=1)
prediction = np.squeeze(prediction,axis=1)
correct_prediction = np.squeeze(correct_prediction,axis=1)
np.save(root+'correct_prediction.npy',correct_prediction)


# ==========  compute the pearson's corrlation coefication between age gap and true age   ========== #
print('=========================')
print('Before bias correction CC',np.corrcoef(target,predicted_age_difference)[0][1])
print('After bias correction CC',np.corrcoef(target,correct_gap)[0][1])

# ==========  compute the MAE between predicted age and true age   ========== #
print('=========================')
print('Before MAE',mean_absolute_error(target,prediction))
print('After correction MAE',mean_absolute_error(target,correct_prediction))

# ==========  compute the spearman's rank corrlation coefication between age gap and true age   ========== #
print('=========================')
print('Before correction sprearman',spearmanr(predicted_age_difference,target,axis=1))
print('After correction sprearman ',spearmanr(correct_gap,target,axis=1))



















# ==========  delet the outlier point  ========== #
# target = np.delete(target,[634,781])
# correct_gap = np.delete(correct_gap,[634,781])
# predicted_age_difference = np.delete(predicted_age_difference,[634,781])

# ==========  plot scatter plot of age gap against wit true age   ========== #
# plt.figure()
# z = np.polyfit(target, correct_gap, 1)
# p = np.poly1d(z)
# plt.plot(target,p(target),color='red',linestyle='-',linewidth=3,
# label='output fit line')
# plt.scatter(target,correct_gap,color='green')
# plt.xlabel('Chronological Age (years)',fontsize=12)
# plt.ylabel('Brain age gap (years)',fontsize=12)
# plt.text(95,15,'MAE = 2.428\nPCC = -0.196\n SRCC = -0.267',
#         horizontalalignment='right',
#         verticalalignment='top')
# plt.title('Results of delta with correction')
# plt.show()