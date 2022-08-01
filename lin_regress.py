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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance

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

def reg_metrics(y, y_pred, weight=1):
    mse = mean_squared_error(y, y_pred)*weight # root mean squared error
    rmse = mean_squared_error(y, y_pred, squared = False)*weight
    r2 = r2_score(y, y_pred)*weight
    adj = adj_r2(r2)
    return mse, rmse, r2, adj

def adj_r2(r2):
    adj = 1 - (1 - r2)*((sample_size - 1)/(sample_size - n_features))
    return adj

def metric_means(estimator, X):
        
    ## kfold cv for r2 scores
    kf = KFold(n_splits = 10, random_state = rand, shuffle = True)
    # skf = StratifiedKFold(n_splits = 10)

    means_train = []
    means_test = []
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = estimator.fit(X_train, y_train)

        y_train_pred = estimator.predict(X[train_index])
        y_test_pred = estimator.predict(X[test_index])

        weight = len(X_train)
        total_w = len(X)

        # performance metrics
        means_train.append(reg_metrics(y_train, y_train_pred, weight))
        means_test.append(reg_metrics(y_test, y_test_pred, weight))

    # weighted mean metrics
    mses_mean_train = np.mean(means_train, axis = 0)[0]/total_w
    rmses_mean_train =  np.mean(means_train, axis = 0)[1]/total_w
    r2s_mean_train = np.mean(means_train, axis = 0)[2]/total_w
    adj_r2s_mean_train = np.mean(means_train, axis = 0)[3]/total_w

    mses_mean_test = np.mean(means_test, axis = 0)[0]/total_w
    rmses_mean_test =  np.mean(means_test, axis = 0)[1]/total_w
    r2s_mean_test = np.mean(means_test, axis = 0)[2]/total_w
    adj_r2s_mean_test = np.mean(means_test, axis = 0)[3]/total_w

    # print metrics
    print('----------------------------------------------')
    print('METRICS')
    print('----------------------------------------------')
    print('Mean R^2 Train: %0.5f' % r2s_mean_train)
    print('Mean R^2 Test: %0.5f' % r2s_mean_test)
    print('----------------------------------------------')
    print('Mean Adj R^2 Train: %0.5f' % adj_r2s_mean_train)
    print('Mean Adj R^2 Test: %0.5f' % adj_r2s_mean_test)
    print('----------------------------------------------')
    print('RMSE Train: %0.5f' % rmses_mean_train)
    print('RMSE Test: %0.5f' % rmses_mean_test)
    print('----------------------------------------------')

    return 

def perm_imp(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rand)

    model = pl.fit(X_train, y_train)

    # low cvs threfore apply permutation importance
    p_imp = permutation_importance(model, X_test, y_test, n_repeats = 30, random_state = rand, scoring = 'r2')
    sorted_idx = p_imp.importances_mean.argsort()
    # print(p_imp)
    for i in sorted_idx[::-1]:
        if p_imp.importances_mean[i] - 2 * p_imp.importances_std[i] > 0:
            print(f'{features.columns[i]:<8}' f'{p_imp.importances_mean[i]:.5f}' f' +/- {p_imp.importances_std[i]:.3f}')

    # plt.boxplot(p_imp.importances[])


filename = "data_for_ML.fits" # this file has been reduced based on the criteria in Section 2.1

dat_tab = Table.read(filename, format = 'fits')

df = dat_tab.to_pandas()
# idx = range(0, 5107)
# df.insert(0,'idx', idx)
# df.set_index('idx')

lr = LinearRegression()
sc = StandardScaler()
pl = make_pipeline(sc, lr)

sample_size = len(df)

features = df[['o3', 'ha', 'hb', 'u_g', 'g_r', 'r_i', 'i_z']]
n_features = len(features.columns)
target = df['n2']

X = features.values
y = target.values

# get cross validated metrics
metric_means(pl, X)

# get cvs r2 and permutation importance
mean_cvs = np.mean(cross_val_score(pl, X, y, cv = 10, scoring = 'r2'))

print('----------------------------------------------')
print('MEAN CROSS VAL SCORE R^2: %0.5f' % mean_cvs)
print('----------------------------------------------')

perm_imp(X,y)


#region residuals
# maybe do for different splits do see, but it is not meaningful to do a mean
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rand)

model = pl.fit(X_train, y_train)

y_train_pred = pl.predict(X_train)
y_test_pred = pl.predict(X_test)

res_train = y_train_pred - y_train
res_test = y_test_pred - y_test

# print(sum(np.square(res_train))) variance ???

print('----------------------------------------------')
print('RESIDUALS')
print('----------------------------------------------')
print('Residuals Mean Train: %0.5f' % np.mean(res_train))
print('Residuals Mean Test: %0.5f' % np.mean(res_test))
print('----------------------------------------------')
print('Residuals Std Train: %0.5f (%0.5f)' % (np.std(res_train), np.std(res_train)/max(res_train)))
print('Residuals Std Test: %0.5f (%0.5f)' % (np.std(res_test), np.std(res_test)/max(res_test)))
print('----------------------------------------------')

# with outliers
plt.figure()
plt.hlines(y = 0, xmin = -500, xmax = 8000, color = gry)
plt.scatter(y_train_pred, y_train_pred - y_train, color = yel, alpha = 0.5, label = 'Train', s = 9)
plt.scatter(y_test_pred, y_test_pred - y_test, color = prp, alpha = 0.5, label = 'Test', s = 7, marker = 's')
plt.xlim(-500, 7500)
plt.xlabel('Predicted Values')
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
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
# plt.savefig('./plots/lin_regress/resid.pdf')
#endregion

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

# plt.show()