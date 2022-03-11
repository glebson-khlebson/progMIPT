import sys
import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, initializers
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

epochs = 10
presets = ['magpie', 'matminer', 'deml']
   
def NN(lr = 0.01, mom = 0, drop_rate = 0.1, neur1 = 256, neur2 = 512, neur3 = 64, init = 'he_normal'):
    
    model = tf.keras.models.Sequential([
        layers.Dense(neur1, activation = 'relu', kernel_initializer = init),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Dense(neur2, activation = 'relu', kernel_initializer = init),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Dense(neur1, activation = 'relu', kernel_initializer = init),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Dense(neur3, activation = 'relu', kernel_initializer = init),
        layers.BatchNormalization(),
        layers.Dropout(drop_rate),
        layers.Dense(1, kernel_initializer = 'normal')
    ])
    
    model.compile(
        loss = 'mse',
        optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = mom),
        metrics = [tf.keras.metrics.RootMeanSquaredError()]
    )
    return model

def Optimize(preset):
    data_load = pd.read_csv(preset+'.csv')
    X = data_load.iloc[:, 1:]
    y = data_load.iloc[:, 0]
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.1, random_state = 42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    
    batch_size = [20, 40, 60, 80]
    learn_rate = [0.01, 0.04, 0.07, 0.08, 0.1, 0.2]
    moms = [0.0, 0.2, 0.4, 0.6, 0.8]
    dropout_rate = [0.05, 0.1, 0.15, 0.2, 0.3]
    initializers = ['uniform', 'normal', 'he_uniform', 'he_normal']
    neurons1 = [64, 128, 256]
    neurons2 = [128, 256, 512]
    neurons3 = [32, 64]

    cv_model = KerasRegressor(build_fn = NN, epochs = epochs)
    param_grid = dict(batch_size = batch_size, lr = learn_rate, mom = moms, init = initializers,
                      drop_rate = dropout_rate, neur1 = neurons1, neur2 = neurons2, neur3 = neurons3)
    grid = GridSearchCV(estimator = cv_model, param_grid = param_grid, n_jobs = -1, cv = 5)
    NN_model = grid.fit(X_train, y_train)
    predictions = NN_model.predict(X_test)
    score = np.sqrt(mse(predictions, y_test))
    best_score = NN_model.best_score_
    print(preset+'\n')
    print(f'Best CV score: RMSE = {best_score} ||  Test score (CB): RMSE = {score}\n')
    print(str(NN_model.best_params_))
    file = open('NN_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt', 'a')
    file.write(preset+'\n')
    file.write(f'Best CV score: RMSE = {best_score} ||  Test score (CB): RMSE = {score}\n')
    file.write(str(NN_model.best_params_))
    file.close()
    return

for preset in presets:
    Optimize(preset)