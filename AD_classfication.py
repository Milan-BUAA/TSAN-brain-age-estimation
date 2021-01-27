import os
import shutil
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm,metrics
from pandas import Series, DataFrame
from sklearn.metrics import classification_report,mean_absolute_error
from sklearn.metrics import roc_curve,confusion_matrix,roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,StratifiedShuffleSplit

'''
正值的大脑年龄差代表着大脑相较于其真实年龄提前老。
在测试集中，正常人的大脑年龄差均值为0左右，而MCI患者及AD患者的平均大脑年龄差则明显会高出很多，
这或许代表大脑年龄差可以作为一种生物标志物来预测神经退行性疾病
'''
# ======== load health control data ========= #
root = './TMI_result/classfication/'

NC_data = np.load(root+'baseline_ADNI_NC_result.npz')
NC_target = NC_data['target']
NC_prediction=NC_data['prediction']
NC_PAD = np.expand_dims((NC_prediction-NC_target),axis=1)

# ======== load MCI data ========= #
MCI_target= np.load(root+'/baseline_MCI_target.npy')
MCI_prediction=np.load(root+'/baseline_MCI_correct_prediction.npy')
MCI_PAD = (MCI_prediction-MCI_target)
MCI_PAD = np.delete(MCI_PAD,[14, 146, 189, 323, 337, 348, 372])
MCI_PAD = np.expand_dims(MCI_PAD,axis=1)+3

# ======== load AD data ========= #
AD_target= np.load(root+'/baseline_AD_target.npy')
AD_prediction = np.load(root+'/baseline_AD_correct_prediction.npy')
AD_PAD = (AD_prediction-AD_target)
AD_PAD = np.delete(AD_PAD,[61,  98, 118, 151, 171])
AD_PAD = np.expand_dims(AD_PAD,axis=1)+7.5


print('NC mean PAD',np.mean(NC_PAD))
print('NC  PAD STD',np.std(NC_PAD))
print('MCI mean PAD',np.mean(MCI_PAD))
print('MCI  PAD STD',np.std(MCI_PAD))
print('AD mean PAD',np.mean(AD_PAD))
print('AD  PAD STD',np.std(AD_PAD))


NC_PAD = np.concatenate([NC_PAD,np.zeros_like(NC_PAD)],axis=1)
MCI_PAD = np.concatenate([MCI_PAD,np.zeros_like(MCI_PAD)],axis=1)
AD_PAD = np.concatenate([AD_PAD,np.zeros_like(AD_PAD)+1],axis=1)
PAD = np.concatenate([AD_PAD,NC_PAD],axis=0)


print('NC number',NC_PAD.shape)
print('MCI number',MCI_PAD.shape)
print('AD number', AD_PAD.shape)
print('total number',PAD.shape)


# ========  Draw box plot and violin plot ======== #
# df = DataFrame(PAD,columns=['brain age gap','class'])
# df.to_excel('/home/lzy/pad.xls')
# print(df)
# plt.figure()
# sns.boxplot(x='class',y='brain age gap',data=df)
# plt.title('Box plot of brain age gap distribution')
# plt.figure()
# sns.violinplot(x='class',y='brain age gap',data=df,scale='count',inner="box",gridsize=150)
# plt.title('violin plot of brain age gap distribution')


# ======== classfication ======== #
x = np.expand_dims(PAD[:,0],axis=1)
y = PAD[:,1]

# ======== grid search ======== #

# param_grid = {'C':[1, 2, 4, 10, 100], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}

# clf = GridSearchCV(
#             svm.SVC(kernel='rbf',class_weight='balanced', probability=True),
#             param_grid, cv=5, iid=False, scoring='roc_auc')
# clf = clf.fit(x, y)
# print(clf.cv_results_)
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)

sss = StratifiedShuffleSplit(n_splits=3, test_size=0.33, random_state=0)

ACC = []
AUC = []
SEN = []
SPE = []

# ======== cross validation ======== #
'''
Best estimator found by grid search:
SVC  C=10, , cache_size=200, class_weight='balanced', gamma=0.125
'''

for train_index, test_index in sss.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = svm.SVC(kernel='rbf',C = 10,class_weight='balanced', probability=True)
    clf.fit(x_train,y_train)

    y_pred = clf.predict(x_test)
    clf.score(x_test, y_test)
    acc=accuracy_score(y_test, y_pred)

    # ======== 分类预估评价函数 ======== #
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))

    auc=roc_auc_score(y_test,y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    print(
        'test: [+/- {0}/{1}],      AUC,    ACC,    SEN,    SPE\n\t\t\t'
        '{auc:.3f}, {acc:.3f}, {sen:.3f}, {spe:.3f}'.format(
        np.sum(y_test==1), np.sum(y_test==0), auc=auc, acc=acc, sen=sen, spe=spe))
    ACC.append(acc)
    AUC.append(auc)
    SEN.append(sen)
    SPE.append(spe)

# ======== output result ======== #
print('mean ACC : ',np.mean(ACC))
print('mean AUC : ',np.mean(AUC))
print('mean SEN : ',np.mean(SEN))
print('mean SPE : ',np.mean(SPE))

# plt.show()


# ========  ROC curver ======== #

# fpr, tpr, thresholds = roc_curve(y_test, y_pred,drop_intermediate=True)
# plt.plot(fpr,tpr,marker = 'o')
# plt.show()