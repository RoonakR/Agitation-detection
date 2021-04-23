from keras import Sequential, Model
from keras.layers import Dense, Input, Conv1D, Dropout, Flatten, Reshape
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.optimizers import RMSprop, Adadelta, Adam, Adagrad
from keras.models import load_model
from keras.callbacks import EarlyStopping

import pandas as pd

import itertools
import random

def encoder_2Dmodel(input_layer, n_features, dr, pooling='max', kr_size=3):
    #encoder
    conv1 = Conv2D(filters=32, kernel_size=kr_size, 
                   strides=1, activation='relu', 
                   padding='same')(input_layer)
    if pooling == 'max':
        pool1 = MaxPooling2D(pool_size=2, strides=2)(conv1)
    elif pooling == 'mean':
        pool1 = AveragePooling2D(pool_size=2, strides=1)(conv1)
    if dr != 0:
        pool1 = Dropout(dr)(pool1)
    
    conv2 = Conv2D(filters=64, kernel_size=kr_size, 
                   strides=1, activation='relu', 
                   padding='same')(pool1)
    if dr != 0:
        conv2 = Dropout(dr)(conv2)
    
    conv3 = Conv2D(filters=128, kernel_size=kr_size, 
                   strides=1, activation='relu', 
                   padding='same')(conv2)
    if dr != 0:
        conv3 = Dropout(dr)(conv3)
    
    flatter = Flatten()(conv3)
    dense1 = Dense(n_features, activation='relu')(flatter)
    
    return dense1


def decoder_2Dmodel(dense1, dr, kr_size=3, act='sigmoid', n_filter=3):
    #deconder
    # for mnist
    # dense2 = Dense(128*14*14)(dense1)
    # reshape = Reshape((14, 14, 128))(dense2)

    dense2 = Dense(128*4*4)(dense1)
    reshape = Reshape((4, 4, 128))(dense2)
    
    conv4 = Conv2D(filters=128, kernel_size=kr_size, 
                   strides=1, activation='relu', 
                   padding='same')(reshape)
    if dr != 0:
        conv4 = Dropout(dr)(conv4)
    
    conv5 = Conv2D(filters=64, kernel_size=kr_size, 
                   strides=1, activation='relu', 
                   padding='same')(conv4)
    up1 = UpSampling2D(size=2)(conv5)
    if dr != 0:
        up1 = Dropout(dr)(up1)
    
    conv6 = Conv2D(filters=32, kernel_size=kr_size, 
                   strides=1, activation='relu', 
                   padding='same')(up1)
    if dr != 0:
        conv6 = Dropout(dr)(conv6)
    
    conv7 = Conv2D(filters=n_filter, kernel_size=kr_size, 
                   strides=1, activation=act, 
                   padding='same', name='autoencoder')(conv6)
    
    return conv7


def get_2Dmodel(n_filter, n_attrib, lr, n_features, dr, pooling='max', act='sigmoid', summary=False, kr_size=3, opt='RMSprop', loss='mean_squared_error'):
    input_shape = (n_attrib, n_attrib, n_filter)
    input_layer = Input(shape=input_shape)

    encoded = encoder_2Dmodel(input_layer, n_features, dr, pooling, kr_size)
    decoded = decoder_2Dmodel(encoded, dr, kr_size, act, n_filter)

    autoencoder = Model(input_layer, decoded)
    if opt == 'RMSprop':
        autoencoder.compile(loss=loss, optimizer=RMSprop(lr=lr), metrics=['mean_squared_error'])
    elif opt == 'Adam':
        autoencoder.compile(loss=loss, optimizer=Adam(lr=lr), metrics=['mean_squared_error'])
    elif opt == 'Adagrad':
        autoencoder.compile(loss=loss, optimizer=Adagrad(lr=lr), metrics=['mean_squared_error']) 
    else:
        autoencoder.compile(loss=loss, optimizer=Adadelta(lr=lr), metrics=['mean_squared_error'])
    if summary:
        autoencoder.summary()

    return input_layer, encoded, autoencoder


def objective(hyperparameters, iteration, fixedparameters, x_train, x_val):
    input_layer, encoded, autoencoder = get_2Dmodel(n_filter=fixedparameters['n_filter'], 
    	n_attrib=fixedparameters['n_attrib'], lr=hyperparameters['lr'], 
    	n_features=hyperparameters['n_features'], dr=hyperparameters['dr'], pooling=hyperparameters['pooling'], 
    	act='sigmoid', summary=False, kr_size=3, opt=hyperparameters['optim'], 
    	loss='mean_squared_error')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    history = autoencoder.fit(x_train, x_train, batch_size=32, epochs=500, 
                          validation_data=(x_val, x_val), verbose=0, callbacks=[es])
    loss = autoencoder.evaluate(x_val, x_val)
    encoder = Model(input_layer, encoded)
    return [loss, hyperparameters, iteration], [history, autoencoder, encoder]

def random_search(param_grid, fixedparameters, x_train, x_val, max_evals):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = ['loss', 'params', 'iteration'],
                                  index = list(range(max_evals)))
    models = pd.DataFrame(columns = ['history', 'autoencoder', 'encoder'],
                                  index = list(range(max_evals)))
    
    # Keep searching until reach max evaluations
    for i in range(max_evals):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

        # Evaluate randomly selected hyperparameters
        eval_results, model = objective(hyperparameters, i, fixedparameters, x_train, x_val)
        
        results.loc[i, :] = eval_results
        models.loc[i, :] = model
    
    # Sort with best score on top
    #results.sort_values('loss', ascending = True, inplace = True)
    #results.reset_index(inplace = True)
    return results , models


def autoencoder_training(params, fixed_params, x_train, x_val):
    results = pd.DataFrame(columns = ['loss', 'params', 'iteration'],
                              index = list(range(max_evals)))
    models = pd.DataFrame(columns = ['history', 'autoencoder', 'encoder'],
                                  index = list(range(max_evals)))
    
    # Keep searching until reach max evaluations
    for idx, param in enumerate(params):

        # Evaluate randomly selected hyperparameters
        eval_results, model = objective(param, idx, fixed_params, x_train, x_val)
        
        results.loc[i, :] = eval_results
        models.loc[i, :] = model
    
    return results , models