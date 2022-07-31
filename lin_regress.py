import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.cosmology as cosmo
from astropy.io import fits
from astropy.table import Table
from mlxtend.plotting import scatterplotmatrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

#################
# colour scheme #
#################
grn ='#54AA68' # comp
blu = '#4B73B2' # sfg
ong = '#DE8551' # agn
gry = '#8D8D8D'
yel = '#CDBA75' # train
prp = '#8273B4' # test

# random state (for consistency)
rand = 4

def reg_metrics(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    adj = adj_r2(r2)
    return mse, r2, adj

def adj_r2(r2):
    adj = 1 - (1 - r2)*((sample_size - 1)/(sample_size - n_features))
    return adj

filename = "data_for_ML.fits" # this file has been reduced based on the criteria in Section 2.1

dat_tab = Table.read(filename, format = 'fits')

df = dat_tab.to_pandas()
# idx = range(0, 5107)
# df.insert(0,'idx', idx)
# df.set_index('idx')
print(df)

sample_size = len(df)

features = df[['o3','ha','hb']]
n_features = len(features.columns)
target = df['n2']

X = features
y = target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rand)

lr = LinearRegression()

#region : learning curve
# # learning curve
# train_sizes, train_scores, test_scores = learning_curve(lr, X, y, train_sizes =  np.linspace(0.1, 1, 10), cv = 10)
# train_mean = np.mean(train_scores, axis = 1)
# train_std = np.std(train_scores, axis = 1)
# test_mean = np.mean(test_scores, axis = 1)
# test_std = np.std(test_scores, axis = 1)

# # plot
# plt.plot(train_sizes, train_mean, color = yel, marker = 'o', label = 'Train')
# plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.5, color = yel)

# plt.plot(train_sizes, test_mean, color = prp, marker = 's', label = 'Test')
# plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.5, color = prp)

# plt.xlabel('Training Size')
# plt.ylabel('Accuracy')

# plt.legend()
#endregion

lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# performance metrics
mse_train, r2_train, adj_r2_train = reg_metrics(y_train, y_train_pred)
mse_test, r2_test, adj_r2_test = reg_metrics(y_test, y_test_pred)
cvs_r2 = np.mean(cross_val_score(lr, X, y, cv = 10, scoring = 'r2'))
# not sure this works because it will fit different models to the test and train
# cvs_r2_train = np.mean(cross_val_score(lr, X_train, y_train, cv = 10, scoring = 'r2'))
# cvs_r2_test = np.mean(cross_val_score(lr, X_test, y_test, cv = 10, scoring = 'r2'))

# print('------------------------------')
# print('MSE Train: %0.3f' % mse_train)
# print('MSE Test: %0.3f' % mse_test)
# print('------------------------------')

print('------------------------------')
print('R^2 Train: %0.5f' % r2_train)
print('R^2 Test: %0.5f' % r2_test)
print('------------------------------')

print('------------------------------')
print('Adjusted R^2 Train: %0.5f' % adj_r2_train)
print('Adjusted R^2 Test: %0.5f' % adj_r2_test)
print('------------------------------')

print('------------------------------')
print('Mean cross_val_score R^2 Train: %0.5f' % cvs_r2)
print('------------------------------')

# print('------------------------------')
# print('Mean cross_val_score R^2 Train: %0.5f' % cvs_r2_train)
# print('Mean cross_val_score R^2 Test: %0.5f' % cvs_r2_test)
# print('------------------------------')

## kfold cv for r2 scores
kf = KFold(n_splits = 10, random_state = rand, shuffle = True)
# skf = StratifiedKFold(n_splits = 10)

r2s_train = []
r2s_test = []
adj_r2s_train = []
adj_r2s_test = []
train_test = []
test_test = []
for train_index, test_index in kf.split(X):

    print(type(train_index))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    lr.fit(X_train, y_train)

    y_train_pred = lr.predict(X[train_index])
    y_test_pred = lr.predict(X[test_index])

    # performance metrics
    mse_train, r2_train, adj_r2_train = reg_metrics(y_train, y_train_pred)
    mse_test, r2_test, adj_r2_test = reg_metrics(y_test, y_test_pred)

    r2s_train.append(r2_train)
    r2s_test.append(r2_test)
    adj_r2s_train.append(adj_r2_train)
    adj_r2s_test.append(adj_r2_test)

    train_test.append(reg_metrics(y_train, y_train_pred))
    test_test.append(reg_metrics(y_test, y_test_pred))

r2s_mean_train = np.mean(np.array(r2s_train))


#region residuals plots
# with outliers
plt.figure()
plt.hlines(y = 0, xmin = -500, xmax = 8000, color = gry)
plt.scatter(y_train_pred, y_train_pred - y_train, color = yel, alpha = 0.5, label = 'Train', s = 9)
plt.scatter(y_test_pred, y_test_pred - y_test, color = prp, alpha = 0.5, label = 'Test', s = 7, marker = 's')
plt.xlim(-500, 7500)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend()
# plt.savefig('./plots/lin_regress/resid_outliers.pdf')

# without (some) outliers
plt.figure()
plt.hlines(y = 0, xmin = -500, xmax = 2100, color = gry)
plt.scatter(y_train_pred, y_train_pred - y_train, color = yel, alpha = 0.5, label = 'Train', s = 9)
plt.scatter(y_test_pred, y_test_pred - y_test, color = prp, alpha = 0.5, label = 'Test', s = 7, marker = 's')
plt.xlim(-15, 399)
plt.ylim(-480,225)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend()
# plt.savefig('./plots/lin_regress/resid.pdf')
#endregion


# plt.show()