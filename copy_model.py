#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.compose import ColumnTransformer
# from sklearn.multiclass import OneVsOneClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from datetime import datetime

pd.options.display.max_columns
pd.set_option('display.max_columns', None)


# # Data Prep

# In[109]:


X_train = pd.read_csv('https://raw.githubusercontent.com/mkleinbort/Kaggle-COMPAS/main/train/X_train.csv', index_col='id')
y_train = pd.read_csv('https://raw.githubusercontent.com/mkleinbort/Kaggle-COMPAS/main/train/y_train.csv', squeeze=True)
X_test = pd.read_csv ('https://raw.githubusercontent.com/mkleinbort/Kaggle-COMPAS/main/test/X_test.csv')


# In[110]:


train_df = X_train
train_df['target'] = y_train
train_df.shape


# In[111]:


train_df['all_priors'] = train_df[[
    'juv_fel_count',
    'juv_misd_count', 
    'juv_other_count',
    'priors_count'
]].sum(axis=1)


# ### Drop redundant columns

# In[112]:



train_df = train_df.drop([
    'name', 'first', 'last',
    'date_of_birth', 'age',
    'juv_fel_count',
    'juv_misd_count',
    'juv_other_count',
    'priors_count'], axis=1)


# ### Deal with date dependent features

# In[113]:


train_df['c_jail_out'] = pd.to_datetime(train_df.c_jail_out)
train_df['c_jail_in'] = pd.to_datetime(train_df.c_jail_in)

train_df['c_jail_days'] = (train_df.c_jail_out - train_df.c_jail_in)
train_df['c_jail_days'] = train_df['c_jail_days'] // np.timedelta64(1, "D")

# remove rows where c_jail_days is null
# train_df.dropna(subset=["c_jail_days"], inplace=True)
# train_df['c_jail_days'].fillna(0, inplace=True)
# train_df = train_df[train_df['c_jail_days'] >= 0]

# How many days_until_rec
train_df['r_jail_in'] = pd.to_datetime(train_df.r_jail_in)

train_df['days_until_rec'] = (train_df.r_jail_in - train_df.c_jail_out)
train_df['days_until_rec'] = train_df['days_until_rec'] // np.timedelta64(1, "D")


# train_df = train_df[train_df['days_until_rec'] >= 0]

# see if the subject has previously recided
train_df['has_r_jail_in'] = train_df['r_jail_in'].notna()

# check if the are out after receeding
# train_df['has_r_jail_out'] = train_df['r_jail_out'].notna()

# calculate number of days in custody - assumption here is that the time spent in custody could lead to recidivism
train_df['out_custody'] = pd.to_datetime(train_df.out_custody)
train_df['in_custody'] = pd.to_datetime(train_df.in_custody)

train_df['custody_days'] = (train_df.out_custody - train_df.in_custody)
train_df['custody_days'] = train_df['custody_days'] // np.timedelta64(1, "D")

# train_df['custody_days'].fillna(0, inplace=True)

# drop more date columns
train_df = train_df.drop([
    "c_jail_out", "out_custody", "in_custody",
    "c_jail_in", "type_of_assessment", "c_offense_date", 
    "r_jail_out", "r_jail_in",
    "screening_date", "v_screening_date",
    "v_type_of_assessment", "c_arrest_date", 
    'c_charge_desc', 
], axis=1)

# train_df.dropna(subset=["target"], inplace=True)


# In[114]:


train_df['target_int'] = train_df['target'].replace({'No-Recidivism': 0, 'Non-Violent': 1, 'Violent': 2})
train_df = train_df.fillna(0)
train_df


# In[115]:


training_features = [
    'sex', 'age_group', 'race', 
    'days_b_screening_arrest',
    'c_charge_degree',
    'days_until_rec',
    'start', 'all_priors',
    'c_jail_days',
    'has_r_jail_in',
#     'has_r_jail_out',
    'custody_days'
]

cat_features = [
    'sex', 
    "age_group",
    'race',
    'c_charge_degree',
    'has_r_jail_in',
#     'has_r_jail_out'
]

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(10, 5), dpi=80)

plt.bar(train_df.index, train_df.days_b_screening_arrest)
plt.ylabel('days_b_screening_arrest')
plt.show()


# # Model

# In[118]:


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

X_split = train_df
y_split = train_df.race
epoch = 0

for train_idx, val_idx in rskf.split(X_split, y_split):

    X_train = X_split.iloc[train_idx].reset_index(drop=True)
    X_val = X_split.iloc[val_idx].reset_index(drop=True)

    training_set = X_train[training_features]
    training_target = X_train.target_int

    validation_set = X_val[training_features]
    validation_target = X_val.target_int

    # encode dataset
    encoder = ColumnTransformer(
        [("OneHotEncoder", OneHotEncoder(), cat_features)],
        remainder='passthrough'
    ).fit(training_set)
    
    training_set = encoder.transform(training_set)
    validation_set = encoder.transform(validation_set)

#     normalize dataset
    normalizer = Normalizer().fit(training_set)
    
    training_set = normalizer.transform(training_set)
    validation_set = normalizer.transform(validation_set)

    # fit model
#     model = OneVsOneClassifier(
    model = LogisticRegressionCV(cv=6, random_state=0, max_iter=10000)
        # LinearSVC(random_state=0, max_iter=10000)
#     )
    
    model.fit(training_set, training_target)

    prediction = model.predict(validation_set)
    prediction_proba = model.predict_proba(validation_set)
#     decision_func = model.decision_function(validation_set)
    score = model.score(training_set, training_target)
    roc = roc_auc_score(validation_target, prediction_proba, multi_class="ovo")
    

    epoch += 1
    print(f"epoch: {epoch} - score: {score:.4f} - roc: {roc:.4f}")

print(np.round(prediction_proba, decimals=4))


# In[119]:


import pickle

pkl_filename = 'logistic_reg_cv.pkl'
logistic_reg_cv_model_pkl = open(pkl_filename, 'wb')
pickle.dump(model, logistic_reg_cv_model_pkl)

logistic_reg_cv_model_pkl.close()


# In[120]:


model.predict(X_test)

