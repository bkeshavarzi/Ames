#import numpy as np
import numpy as np
import pandas as pd
import sklearn
import statistics

import warnings
warnings.filterwarnings('ignore')

housing = pd.read_csv('AmesHousing.csv')
housing = pd.read_csv('AmesHousing.csv')
new_col = [i.lower().replace(' ','_').replace('/','_') for i in housing.columns]
housing.columns = new_col
housing_copy = housing.copy()

#Modyfing data frame

housing_copy['age'] = housing_copy.yr_sold - housing_copy.year_built
housing_copy['remodeled_age'] = housing_copy.yr_sold - housing_copy['year_remod_add']
housing_copy['garage_age'] = housing_copy.yr_sold - housing_copy['garage_yr_blt']

housing_copy.drop(['order','pid'],axis=1,inplace=True)
housing_copy.drop(['year_built'],axis=1,inplace=True)
housing_copy.drop(['year_remod_add'],axis=1,inplace=True)
housing_copy.drop(['garage_yr_blt'],axis=1,inplace=True)
housing_copy.drop(['pool_qc'],axis=1,inplace=True)
housing_copy.drop(['fence'],axis=1,inplace=True)
housing_copy.drop(['misc_feature'],axis=1,inplace=True)
housing_copy.drop(['misc_val'],axis=1,inplace=True)
housing_copy.drop(['yr_sold'],axis=1,inplace=True)
housing_copy.drop(['alley'],axis=1,inplace=True)



housing_copy.to_csv('HousingCopy.csv')


