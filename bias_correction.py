import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr,spearmanr


# ==========-   load  prediction and target  ========== #
# Note: predicition brain age and target age are from training set.
train_root = './trainset_result/'
train_target = np.load(os.path.join(train_root,'target.npy'))
train_prediction = np.load(os.path.join(train_root,'prediction.npy'))
# ==========   reshape numpy array   ========== #
train_prediction = np.expand_dims(train_prediction,axis=1)
train_target = np.expand_dims(train_target,axis=1)

# ========== fit linear regression function === #
reg = LinearRegression()
reg.fit(train_target,train_prediction-train_target)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

# ==========-   load  prediction and target  ========== #
# Note: Use linear regression parameter training from training set
#       to do bias correction for test set.
test_root = './testset_result/'
test_target = np.expand_dims(np.load(os.path.join(test_root,'target.npy')),axis=1)
test_prediction = np.expand_dims(np.load(os.path.join(test_root,'prediction.npy')),axis=1)
predicted_age_difference = test_prediction-test_target
print('Number of Test Sample: ',predicted_age_difference.shape[0])
# ==========   bias correction   ========== #
correct_prediction = test_prediction - (reg.coef_[0][0]*test_target + reg.intercept_[0])
corrected_age_difference = correct_prediction - test_target
corrected_age_difference = np.squeeze(corrected_age_difference,axis=1)

# ==========   reshape the corrected brain age numpy array and save it   ========== #
test_target = np.squeeze(test_target,axis=1)
test_prediction = np.squeeze(test_prediction,axis=1)
correct_prediction = np.squeeze(correct_prediction,axis=1)
np.save(test_root+'correct_prediction.npy',correct_prediction)

# ==========  compute the pearson's corrlation coefication between age gap and true age   ========== #
print('=========================')
print('Before bias correction CC',pearsonr(test_target,predicted_age_difference)[0][1])
print('After bias correction CC',pearsonr(test_target,corrected_age_difference)[0][1])

# ==========  compute the MAE between predicted age and true age   ========== #
print('=========================')
print('Before MAE',mean_absolute_error(test_target,test_prediction))
print('After correction MAE',mean_absolute_error(test_target,correct_prediction))

# ==========  compute the spearman's rank corrlation coefication between age gap and true age   ========== #
print('=========================')
print('Before correction sprearman',spearmanr(predicted_age_difference,test_target,axis=1))
print('After correction sprearman ',spearmanr(corrected_age_difference,test_target,axis=1))



















# ==========  delet the outlier point  ========== #
# target = np.delete(target,[634,781])
# corrected_age_difference = np.delete(corrected_age_difference,[634,781])
# predicted_age_difference = np.delete(predicted_age_difference,[634,781])

# ==========  plot scatter plot of age gap against wit true age   ========== #
# plt.figure()
# z = np.polyfit(target, corrected_age_difference, 1)
# p = np.poly1d(z)
# plt.plot(target,p(target),color='red',linestyle='-',linewidth=3,
# label='output fit line')
# plt.scatter(target,corrected_age_difference,color='green')
# plt.xlabel('Chronological Age (years)',fontsize=12)
# plt.ylabel('Brain age gap (years)',fontsize=12)
# plt.text(95,15,'MAE = 2.428\nPCC = -0.196\n SRCC = -0.267',
#         horizontalalignment='right',
#         verticalalignment='top')
# plt.title('Results of delta with correction')
# plt.show()