import matminer
import pandas as pd
import numpy as np
from matminer.datasets import get_available_datasets, load_dataset
import sys
import datetime

df = load_dataset('steel_strength')

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras import layers, initializers
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from catboost import CatBoostRegressor, Pool, cv
from matminer.featurizers.composition import ElementProperty
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.decomposition import PCA

df_ess = df[[df.columns[0], df.columns[15]]]
from matminer.featurizers.conversions import StrToComposition
stc = StrToComposition()
df_form = stc.featurize_dataframe(df_ess, 'formula')


def CB(X_train, X_test, y_train, y_test):
    pipeline_cb = Pipeline([
        ('scaler3', StandardScaler()),
        ('pca3', PCA()),
        ('regressor3', CatBoostRegressor())
    ])
    parameters_cb = {
        'pca3__n_components': np.arange(10, 30, 5),
        'regressor3__iterations': [1000, 1100, 1200, 1300],
        'regressor3__learning_rate': [0.04, 0.06, 0.08, 0.09, 0.1, 0.12],
        'regressor3__depth': [4, 5, 6, 8]
    }
    gridsearch_cb = GridSearchCV(pipeline_cb, parameters_cb, scoring = 'neg_mean_squared_error', cv = 5)
    cb_model = gridsearch_cb.fit(X_train, y_train)
    predictions = cb_model.predict(X_test)
    score = np.sqrt(mse(predictions, y_test))
    best_score_CB = cb_model.best_score_
    print(f'Best CV score: RMSE = {np.sqrt(-best_score_CB)}  Test score (CB): RMSE = {score}')
    best_params_CB = cb_model.best_params_
    print(best_params_CB)
    print(cb_model.best_estimator_)
    file = open('CB_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt', 'a')
    file.write(best_params_CB)
    file.write('\n')
    file.write(y_test)
    file.write('\n')
    file.write(predictions)
    file.write('\n')
    file.close()
    return 
    
def GB(X_train, X_test, y_train, y_test):
    pipeline_gb = Pipeline([
        ('scaler2', StandardScaler()),
        ('pca2', PCA()),
        ('regressor2', GradientBoostingRegressor())
    ])
    parameters_gb = {
        'pca2__n_components': np.arange(10, 30, 5),
        'regressor2__n_estimators': [100, 150, 200, 250, 300, 400],
        'regressor2__learning_rate': [0.05, 0.06, 0.08, 0.1, 0.12],
        'regressor2__min_samples_split':[3, 4, 5, 6, 8],
        'regressor2__max_features':['auto', 'sqrt', 'log2'],
        'regressor2__max_depth':[20, 25, 40],
        'regressor2__min_samples_leaf':[1,2,3,4,5],
        'regressor2__max_leaf_nodes':[10, 20, 30, None]
    }
    gridsearch_gb = GridSearchCV(pipeline_gb, parameters_gb, scoring = 'neg_mean_squared_error', cv = 5)
    gb_model = gridsearch_gb.fit(X_train, y_train)
    predictions = gb_model.predict(X_test)
    score = np.sqrt(mse(predictions, y_test))
    best_score_GB = gb_model.best_score_
    print(f'Best CV score: RMSE = {np.sqrt(-best_score_GB)}  Test score (GB): RMSE = {score}')
    best_params_GB = gb_model.best_params_
    print(best_params_GB)
    print(gb_model.best_estimator_)
    file = open('GB_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt', 'a')
    file.write(best_params_GB)
    file.write('\n')
    file.write(y_test)
    file.write('\n')
    file.write(predictions)
    file.write('\n')
    file.close()
    return

def RF(X_train, X_test, y_train, y_test):
    
    pipeline_rf = Pipeline([
        ('scaler1', StandardScaler()),
        ('pca1', PCA()),
        ('regressor1', RandomForestRegressor(random_state = 42))
    ])
    parameters_rf = {
        'pca1__n_components': np.arange(10, 30, 5),
        'regressor1__n_estimators': [100, 150, 200, 300, 400, 500],
        'regressor1__min_samples_split':[2, 4, 6],
        'regressor1__max_features':['auto', 'sqrt', 'log2'],
        'regressor1__max_depth':[20, 30, 40, None],
        'regressor1__min_samples_leaf':[1,2,3,4],
        'regressor1__max_leaf_nodes':[10, 20, None]
    }
    gridsearch_rf = GridSearchCV(pipeline_rf, parameters_rf, scoring = 'neg_mean_squared_error', cv = 5)
    rf_model = gridsearch_rf.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    score = np.sqrt(mse(predictions, y_test))
    best_score_RF = rf_model.best_score_
    print(f'Best CV score: RMSE = {np.sqrt(-best_score_RF)}  Test score (RF): RMSE = {score}')
    best_params_RF = rf_model.best_params_
    print(best_params_RF)
    print(rf_model.best_estimator_)
    file = open('RF_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt', 'a')
    file.write(best_params_RF)
    file.write('\n')
    file.write(y_test)
    file.write('\n')
    file.write(predictions)
    file.write('\n')
    file.close()
    return     

def Selection(preset, Func):
    ep = ElementProperty.from_preset(preset)
    df_preset = ep.featurize_dataframe(df_form, ['composition'])
    df_preset = df_preset.drop(['formula', 'composition'], axis=1)
    df_preset = df_preset.loc[:, (df_preset != 0).any(axis=0)] 
    df_preset.dropna(axis=1, inplace=True) 
    df_preset = df_preset.loc[:, (df_preset != df_preset.iloc[0]).any()] 
    df_preset = pd.get_dummies(df_preset)
    X = df_preset[df_preset.columns[1:]]
    y = df_preset['tensile strength']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.1, random_state = 42)
    Func(X_train, X_test, y_train, y_test)
    return

Jobs = [[CB, 'magpie'], [CB, 'matminer'], [CB, 'deml'],
        [GB, 'magpie'], [GB, 'matminer'], [GB, 'deml'],
        [RF, 'magpie'], [RF, 'matminer'], [RF, 'deml']]
Function = Jobs[sys.argv[1]][0]
preset = Jobs[sys.argv[1]][1]
print(f'Studying preset {preset}...')
Selection(preset, Function)
