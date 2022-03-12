import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

epochs = 80
presets = ['magpie', 'matminer', 'deml']
   
def NN(neur2 = 128, neur1 = 128, neur3 = 64, drop_rate = 0.1, mom = 0.8, lr = 0.08,
       init = 'he_normal', l1 = 1e-5, l2 = 1e-5):  
    
    
    
    model = tf.keras.models.Sequential([
        layers.Dense(neur1, activation = 'relu', kernel_initializer = init, 
                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), bias_regularizer=regularizers.l2(l2),
                     activity_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Dense(neur2, activation = 'relu', kernel_initializer = init,
                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), bias_regularizer=regularizers.l2(l2),
                     activity_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Dense(neur2, activation = 'relu', kernel_initializer = init, 
                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), bias_regularizer=regularizers.l2(l2),
                     activity_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Dense(neur2, activation = 'relu', kernel_initializer = init, 
                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), bias_regularizer=regularizers.l2(l2),
                     activity_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Dense(neur3, activation = 'relu', kernel_initializer = init, 
                    kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), bias_regularizer=regularizers.l2(l2),
                     activity_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Dense(1, activation = 'linear', kernel_initializer = 'normal')
    ])
    
    model.compile(
        loss = 'mse',
        optimizer = tf.keras.optimizers.SGD(lr = lr, momentum = mom),
        metrics = [tf.keras.metrics.RootMeanSquaredError()]
    )
    return model

def Optimize(preset):
    data_load = pd.read_csv(preset+'.csv')
    data_load = data_load.loc[:, (data_load != 0).any(axis=0)] 
    data_load.dropna(axis=1, inplace=True) 
    data_load = data_load.loc[:, (data_load != data_load.iloc[0]).any()] 

    X = data_load.iloc[:, 2:].to_numpy()
    y = data_load.iloc[:, 1].to_numpy()
    scaler = StandardScaler()
    scaler_y = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.1, random_state = 42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(len(y_train),1))[:,0]
    y_test = scaler_y.transform(y_test.reshape(len(y_test),1))[:,0]

    
    initializers = ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    learn_rate = [0.01, 0.03, 0.05, 0.07, 0.09]
    moms = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    dropout_rate = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07]
    neurons1 = [32, 64, 128, 256, 512]
    neurons2 = [64, 128, 256, 512]
    neurons3 = [32, 64, 128, 256]
    batch_size = [10, 20, 40, 60, 80]
    l1s = [0.0, 1e-5, 1e-4]
    l2s = [0.0, 1e-5, 1e-4]
    
    cv_model = KerasRegressor(build_fn = NN, epochs = epochs, verbose = 0)
    param_grid = dict(neur1 = neurons1, neur2 = neurons2, neur3 = neurons3, drop_rate = dropout_rate, mom = moms,
                      lr = learn_rate, init = initializers, batch_size = batch_size, l1 = l1s, l2 = l2s)
    grid = GridSearchCV(estimator = cv_model, param_grid = param_grid, scoring = 'neg_mean_squared_error', n_jobs = -1, cv = 5)
    NN_model = grid.fit(X_train, y_train)
    predictions = NN_model.predict(X_test)
    determ = r2(predictions, y_test)
    score = np.sqrt(mse(predictions, y_test))
    best_score = NN_model.best_score_
    print(preset+'\n')
    print(f'Best mean CV score: RMSE = {np.sqrt(-best_score)} || Test score: RMSE = {score}\nDetermination = {determ}')
    print(str(NN_model.best_params_))
    file = open('NN_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt', 'a')
    file.write(preset+'\n')
    file.write(f'Best mean CV score: RMSE = {np.sqrt(-best_score)} || Test score: RMSE = {score}\nDetermination = {determ}')
    file.write(str(NN_model.best_params_))
    file.close()
    return
preset = presets[int(sys.argv[1])]
Optimize(preset)