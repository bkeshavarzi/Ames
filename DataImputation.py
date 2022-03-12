#import numpy as np
import numpy as np
import pandas as pd
import sklearn
import statistics
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

housing = pd.read_csv('housingCopy.csv')
housing_copy = housing.copy()

cat_feature_name  =  []
cat_feature_index =  []
con_feature_name  =  []
con_feature_index =  []
nan_dict = {}
nan_col_index =[]

neg_fun = lambda x : not x
neg_fun_vec = np.vectorize(neg_fun)

from sklearn.preprocessing import OrdinalEncoder
OE = OrdinalEncoder()

for ind,icol in enumerate(housing_copy.columns):

    if (str(housing_copy.loc[:,icol].dtype) == 'object'):

        cat_feature_name.append(icol)
        cat_feature_index.append(ind)
        nan_index = housing_copy.loc[:,icol].isna()

        if (nan_index.sum() != 0):

            mode_data = statistics.mode(housing_copy.loc[neg_fun_vec(nan_index),icol])
            nan_col_index.append(ind)
            housing_copy.loc[nan_index,icol] = mode_data
            nan_dict[icol] = housing_copy.loc[:,icol].isna()

    else:

        con_feature_name.append(icol)
        con_feature_index.append(ind)
        nan_index = housing_copy.loc[:, icol].isna()

        if (nan_index.sum() != 0):

            median_data = housing_copy.loc[neg_fun_vec(nan_index), icol].median()
            nan_col_index.append(ind)
            housing_copy.loc[nan_index, icol] = median_data
            nan_dict[icol] = nan_index

data_array = OE.fit_transform(housing_copy)
housing_copy = pd.DataFrame(data=data_array,index=housing_copy.index,columns=housing_copy.columns)

for ikey in nan_dict.keys():
    housing_copy.loc[nan_dict[ikey],ikey] = np.nan

housing_full = housing_copy.dropna(axis=0,how='any')
full_shape = housing_full.shape

np.random.seed(26)

train_index = list(set(np.random.randint(0,full_shape[0],int(0.6*full_shape[0]))))
Y_true  = housing_full.copy()
X_test  = housing_full.copy()
X_test.iloc[train_index,nan_col_index] = np.nan

#Try to find the best number of neighbors

from sklearn.impute import KNNImputer

n_neighbors_list = np.arange(1,int(full_shape[0]**0.5),5)

MSE = lambda x,y : np.sum((x.values-y.values)**2)**0.5/x.shape[0]

score_df = pd.DataFrame(np.zeros((len(n_neighbors_list),len(X_test.columns))))
score_df.index = n_neighbors_list
score_df.columns = X_test.columns

for i_neighbor in n_neighbors_list:

    knn = KNNImputer()
    knn.set_params(n_neighbors=i_neighbor)
    data = knn.fit_transform(X_test)
    data = OE.inverse_transform(data)
    df = pd.DataFrame(data,index=housing_full.index,columns=housing_full.columns)

    for col_name in df.columns:

        if (col_name in cat_feature_name):

            col_score = np.sum(df.loc[:, col_name].values == Y_true.loc[:, col_name]) *100.0 / df.shape[0]
            score_df.loc[i_neighbor, col_name] = col_score

        elif (col_name in con_feature_name):

            col_score = MSE(df.loc[:, col_name], Y_true.loc[:, col_name])
            score_df.loc[i_neighbor, col_name] = col_score

#Based on the above observation, optimum number of neighbors are 11
knn = KNNImputer()
knn.set_params(n_neighbors = 11)
housing_copy = housing.copy()

for ind,icol in enumerate(housing_copy.columns):
    if (ind not in nan_col_index):
        continue
    if (str(housing_copy.loc[:,icol].dtype)!='object'):
        median_val = np.median(housing_copy.loc[neg_fun_vec(nan_dict[icol]),icol])
        housing_copy.loc[nan_dict[icol], icol] = median_val
    else:
        mode_val = statistics.mode(housing_copy.loc[neg_fun_vec(nan_dict[icol]),icol])
        housing_copy.loc[nan_dict[icol], icol] = mode_val

data = OE.fit_transform(housing_copy)
housing_knn = pd.DataFrame(data,index=housing_copy.index,columns=housing_copy.columns)

for ind,icol in enumerate(housing_copy.columns):
    if (ind not in nan_col_index):
        continue
    housing_knn.loc[nan_dict[icol], icol] = np.nan

data = knn.fit_transform(housing_knn)
data = OE.inverse_transform(data)
housing_knn = pd.DataFrame(data,index=housing_copy.index,columns=housing_copy.columns)

housing_knn.to_csv('housing_KNN.csv')


housing_median_mean = housing.copy()
for ind,icol in enumerate(housing_copy.columns):
    if (ind not in nan_col_index):
        continue
    if (str(housing_copy.loc[:,icol].dtype)!='object'):
        median_val = np.median(housing_copy.loc[neg_fun_vec(nan_dict[icol]),icol])
        housing_median_mean.loc[nan_dict[icol], icol] = median_val
    else:
        mode_val = statistics.mode(housing_copy.loc[neg_fun_vec(nan_dict[icol]),icol])
        housing_median_mean.loc[nan_dict[icol], icol] = mode_val

housing_median_mean.to_csv('Housing_Median_Mean.csv')
print(housing_median_mean.isna().sum())