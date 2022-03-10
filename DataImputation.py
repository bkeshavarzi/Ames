#import numpy as np
import numpy as np
import pandas as pd
import sklearn
import statistics

import warnings
warnings.filterwarnings('ignore')

housing = pd.read_csv('housingCopy.csv')
housing_copy = housing.copy()

cat_feature_name  =  []
cat_feature_index =  []
con_feature_name  =  []
con_feature_index =  []
nan_dict = {}

neg_fun = lambda x : not x
neg_fun_vec = np.vectorize(neg_fun)

from sklearn.preprocessing import OrdinalEncoder
OE = OrdinalEncoder()

for ind,icol in enumerate(housing_copy.columns):

    if (str(housing_copy.loc[:,icol].dtype) == 'object'):

        cat_feature_name.append(icol)
        cat_feature_index.append(ind)
        nan_index = housing_copy.loc[:,icol].isna()

        if (len(nan_index) != 0):

            mode_data = statistics.mode(housing_copy.loc[neg_fun_vec(nan_index),icol])
            housing_copy.loc[nan_index,icol] = mode_data
            nan_dict[icol] = nan_index

    else:

        con_feature_name.append(icol)
        con_feature_index.append(ind)
        nan_index = housing_copy.loc[:, icol].isna()

        if (len(nan_index) != 0):

            median_data = housing_copy.loc[neg_fun_vec(nan_index), icol].median()
            housing_copy.loc[nan_index, icol] = median_data
            nan_dict[icol] = nan_index

data_array = OE.fit_transform(housing_copy)
housing_copy = pd.DataFrame(data=data_array,index=housing_copy.index,columns=housing_copy.columns)

for ikey in nan_dict.keys():
    housing_copy.loc[nan_dict[ikey],ikey] = np.nan

housing_full = housing_copy.dropna(axis=0,how='any')
full_shape = housing_full.shape
print(full_shape[0])
np.random.seed(26)
test_index  = np.random.randint(0,full_shape[0],int(full_shape[0]*0.7))
train_index = [i for i in range(full_shape[0]) if i not in test_index]

print(len(test_index))
print(len(train_index))
#try to find the best number of neighbors

from sklearn.impute import KNNImputer


n_neighbors_list = np.arange(1,int(full_shape**0.5),10)

for i_neighbor in n_neighbors_list:

    knn = KNNImputer()
    knn.set_params(n_neighbors=i_neighbor)
    knn.fit_transform(housing_full.iloc[train_index,:])
