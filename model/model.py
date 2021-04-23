import os
import numpy as np

import keras
from keras import Model
from keras.layers import Dense, Input, Conv1D, Dropout, Flatten, Reshape,\
						MaxPooling1D, AveragePooling1D, BatchNormalization, LSTM,\
						InputLayer, Activation
from keras.models import Sequential
	
from keras.regularizers import l2	
from model.Probability_clf import Probability_CLF_Mul		


def conv_block(input_layer, filter_size, kernel_size, stride, dropout, dropout_rate,
			 batch_norm, activation='relu', padding='same'):
	conv = Conv1D(filters=filter_size, kernel_size=kernel_size, 
                   strides=stride, activation=activation, 
                   padding=padding)(input_layer)
	if batch_norm:
		conv = BatchNormalization()(conv)
	if dropout:
		conv = Dropout(dropout_rate)(conv)

	return conv

def self_supervised_model(input_shape, dropout, dropout_rate, batch_norm, pooling, pooling_type, 
						hidden_nodes = 128, stride_mp=4, n_class=10):
	input_layer = Input(shape=input_shape)

	conv1 = conv_block(input_layer, filter_size=32, kernel_size=32, stride=1, dropout=dropout, 
		dropout_rate=dropout_rate, batch_norm=batch_norm, activation='relu', padding='same')
	conv1 = conv_block(conv1, filter_size=32, kernel_size=32, stride=1, dropout=dropout, 
		dropout_rate=dropout_rate, batch_norm=batch_norm, activation='relu', padding='same')

	if pooling:
		if pooling_type == 'max':
			conv1 = MaxPooling1D(pool_size=8, stride=1, padding='valid')(conv1)
		else:
			conv1 = AveragePooling1D(pool_size=8, stride=1, padding='valid')(conv1)

	conv2 = conv_block(conv1, filter_size=64, kernel_size=16, stride=1, dropout=dropout, 
		dropout_rate=dropout_rate, batch_norm=batch_norm, activation='relu', padding='same')
	conv2 = conv_block(conv2, filter_size=64, kernel_size=16, stride=1, dropout=dropout, 
		dropout_rate=dropout_rate, batch_norm=batch_norm, activation='relu', padding='same')

	if pooling:
		if pooling_type == 'max':
			conv2 = MaxPooling1D(pool_size=8, stride=1, padding='valid')(conv2)
		else:
			conv2 = AveragePooling1D(pool_size=8, stride=1, padding='valid')(conv2)

	conv3 = conv_block(conv2, filter_size=128, kernel_size=8, stride=1, dropout=dropout, 
		dropout_rate=dropout_rate, batch_norm=batch_norm, activation='relu', padding='same')
	conv3 = conv_block(conv3, filter_size=128, kernel_size=8, stride=1, dropout=dropout, 
		dropout_rate=dropout_rate, batch_norm=batch_norm, activation='relu', padding='same')

	if pooling:
		if pooling_type == 'max':
			conv3 = MaxPooling1D(pool_size=conv3.get_shape()[1].value, 
				stride=stride_mp, padding='valid')(conv3)
		else:
			conv3 = AveragePooling1D(pool_size=conv3.get_shape()[1].value, 
				stride=stride_mp, padding='valid')(conv3)

	flat_layer = Flatten()(conv3)
	dense = Dense(hidden_nodes, activation='relu')(flat_layer)
	output = Dense(n_class, activation='sigmoid')(dense)

	model = Model(input_layer, output)
	return model


def classifier_model(input_shape, n_class, hidden_nodes=128, L2=0, batch_norm=False, lstm=False):
	
	input_layer = Input(shape=input_shape)
	hidden_layer = input_layer
	if batch_norm:
		hidden_layer = BatchNormalization()(hidden_layer)
	if lstm:
		hidden_layer = LSTM(hidden_nodes, return_sequences=True)(hidden_layer)
	else:
		hidden_layer = Dense(hidden_nodes, activation='relu', kernel_regularizer = l2(L2))(hidden_layer)
		
	hidden_layer = Dense(hidden_nodes, activation='relu', kernel_regularizer = l2(L2))(hidden_layer)
	hidden_layer = Flatten()(hidden_layer)
	out_layer = Dense(n_class, activation='sigmoid')(hidden_layer)
	model = Model(input_layer, out_layer)

	return model

def pnn_classifier_model(input_shape, n_class=2, n_layer=2, batch_norm=False):
    
    input_layer = Input(shape=input_shape)
    hidden_layer = input_layer
    if batch_norm:
    	hidden_layer = BatchNormalization()(hidden_layer)

    if(n_layer == 1):
    	hidden_layer = Dense(200, activation='relu')(hidden_layer)
    elif(n_layer == 2):
    	hidden_layer = Dense(200, activation='relu')(hidden_layer)
    	hidden_layer = Dense(100, activation='relu')(hidden_layer)
    else:
    	hidden_layer = Dense(200, activation='relu')(hidden_layer)
    	hidden_layer = Dense(100, activation='relu')(hidden_layer)
    	hidden_layer = Dense(50, activation='relu')(hidden_layer)

    hidden_layer = Flatten()(hidden_layer)
    out_layer = Probability_CLF_Mul(n_class)(hidden_layer)
    model = Model(input_layer, out_layer)
  
    return model


def cnn_classifier_model(input_shape, n_class=2, batch_norm=False):
    cnn_model = Sequential()
    cnn_model.add(InputLayer(input_shape=input_shape))
    if batch_norm:
    	cnn_model.add(BatchNormalization())
    cnn_model.add(Conv1D(32, 32, strides=1, padding='same'))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling1D(pool_size=8, stride=1, padding='valid'))
    cnn_model.add(Conv1D(64, 16, strides=1, padding='same'))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling1D(pool_size=8, stride=1, padding='valid'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(n_class))

    return cnn_model