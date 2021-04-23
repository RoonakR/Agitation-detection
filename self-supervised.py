import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from model.autoencoder import autoencoder_training
from utils.utils import make_selfsupervised_dataset
from model.model import self_supervised_model


params_grid = [
	{
		'optim': 'RMSprop',
		'lr' : 0.0001,
		'dr': 0.2,
		'n_features': 43,
		'pooling': 'max',
		'batch_size': 128,
		'n_layer': 2
    },
	{
		'optim': 'Adam',
	    'lr' : 0.003,
	    'dr': 0.4,
	    'n_features': 43,
	    'pooling': 'max',
	    'batch_size': 64,
	    'n_layer': 2
    },
    {
		'optim': 'Adadelta',
	    'lr' : 0.1,
	    'dr': 0.4,
	    'n_features': 43,
	    'pooling': 'max',
	    'batch_size': 32,
	    'n_layer': 3
    },
    {
		'optim': 'Adam',
	    'lr' : 0.0001,
	    'dr': 0.1,
	    'n_features': 32,
	    'pooling': 'mean',
	    'batch_size': 128,
	    'n_layer': 1
    },
    {
		'optim': 'Adagrad',
	    'lr' : 0.003,
	    'dr': 0.3,
	    'n_features': 32,
	    'pooling': 'mean',
	    'batch_size': 32,
	    'n_layer': 1
    },
    {
		'optim': 'RMSprop',
	    'lr' : 0.01,
	    'dr': 0.4,
	    'n_features': 32,
	    'pooling': 'mean',
	    'batch_size': 128,
	    'n_layer': 2
    },
    {
		'optim': 'Adam',
	    'lr' : 0.003,
	    'dr': 0.01,
	    'n_features': 32,
	    'pooling': 'mean',
	    'batch_size': 32,
	    'n_layer': 3
    },
    {
		'optim': 'Adadelta',
	    'lr' : 0.01,
	    'dr': 0.2,
	    'n_features': 24,
	    'pooling': 'max',
	    'batch_size': 128,
	    'n_layer': 1
    },
    {
		'optim': 'RMSprop',
	    'lr' : 0.0005,
	    'dr': 0.1,
	    'n_features': 24,
	    'pooling': 'mean',
	    'batch_size': 64,
	    'n_layer': 3
    },
    {
		'optim': 'Adam',
	    'lr' : 0.1,
	    'dr': 0.1,
	    'n_features': 24,
	    'pooling': 'mean',
	    'batch_size': 64,
	    'n_layer': 3
    }


]

params = {
    'n_attrib': 8,
    'n_filter': 3
}

data_unlabel = np.load('data/data_unlabel_train.pkl', allow_pickle=True)
data_unlabel = np.reshape(data_unlabel, (-1, 8, 8, 3))
data_label = np.load('data/data_label_train.pkl', allow_pickle=True)
data_label = np.reshape(data_label, (-1, 8, 8, 3))

x_train, x_val = train_test_split(data_unlabel, test_size=0.2, random_state=13, shuffle= True)

results, models = autoencoder_training(params_grid, params, x_train, x_val)

for i in range(0, 10):
	print("encoder ", i, ":")
	print(results['loss'][i])

for i in range(0, 10):
    models['autoencoder'][i].save('models/autoencoder'+ str(i+1) + '.hdf5')
    models['encoder'][i].save('models/encoder_'+  str(i+1) + '.h5')


data_trans, label_trans = make_selfsupervised_dataset(models['encoder'], data_unlabel)

x, x_test, y, y_test = train_test_split(data_trans, label_trans, test_size=0.2, random_state=13, 
                                                  shuffle= True, stratify=label_trans)

ss_params={
	'input_shape': (data_trans.shape[1], data_trans.shape[2]),
	'dropout': True,
	'dropout_rate': 0.1,
	'batch_norm': False,
	'pooling': True,
	'pooling_type': 'max',
	'n_class': len(models['encoder'])+1
}

loss = 'binary_crossentropy'
learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)

ss_model = self_supervised_model(ss_params['input_shape'], ss_params['dropout'], 
								ss_params['dropout_rate'], ss_params['batch_norm'], 
								ss_params['pooling'], ss_params['pooling_type'], 
                                hidden_nodes=128, stride_mp=4, n_class=ss_params['n_class'])

ss_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
history = ss_model.fit(x, y, batch_size=32, epochs=50, 
                       validation_data=(x_test, y_test), verbose=1)
test_eval = ss_model.evaluate(x_test, y_test, verbose = 0) 
print("self-supervised evaluation results: ", test_eval)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.ylabel('Loss')
plt.legend()
plt.xlabel('Epochs')

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='valid')
plt.ylabel('Accuracy')
plt.legend()
plt.xlabel('Epochs')

ss_model.save('saved_models/ss_model.hdf5')

