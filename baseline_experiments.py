import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.naive_bayes import GaussianNB
from keras.optimizers import Adam
from utils.evaluation import categorical_evaluation
from model.baseline_models import deepLSTM, deepBiLSTM, ResNet_model, VGG_model, Inception_model

rseed = 13
np.random.seed(rseed)
n_features = 8
timesteps = 24
epochs = 50
params_rf = {'n_estimators': 166,
                 'max_features': 'sqrt',
                 'max_depth': 110,
                 'min_samples_split': 2,
                 'min_samples_leaf': 1,
                 'bootstrap': False}

params_svm = {'C': 100000.0,
                  'gamma': 1e-07,
                  'kernel': 'rbf',
                  'degree': 1,
                  'class_weight': None}

params_gp = {'kernel': 1**2 * DotProduct(sigma_0=1)}

lstm_params = {
    'n_class': 2,
    'n_layer': 2,
    'n_units': 200,
    'drop_out': 0.2,
    'lr': 0.0005623413251903491,
    'loss': 'binary_crossentropy',
    'active': 'relu'
}

bilstm_params = {
    'n_class': 2,
    'n_layer': 1,
    'n_units': 1000,
    'drop_out': 0.1,
    'lr': 0.0031622776601683794,
    'loss': 'binary_crossentropy'
}

res_params = {
    'n_channel': 3,
    'n_class': 2,
    'lr': 0.0031622776601683794,
    'loss': 'binary_crossentropy',
    'n_layers': 3,
    'n_units': 200
}

vgg_params = {
    'n_channel': 3,
    'n_class': 2,
    'lr': 0.0001,
    'loss': 'binary_crossentropy',
    'n_layers': 2,
    'n_units': 1000
}

inc_params = {
    'n_channel': 3,
    'n_class': 2,
    'lr': 0.0001,
    'loss': 'binary_crossentropy',
    'n_layers': 1,
    'n_units': 200
}

data_label =  np.load('data/data_label_train.pkl', allow_pickle=True)
label = np.load('data/label_train.pkl', allow_pickle=True)
data_label = np.reshape(data_label, (-1, timesteps*n_features))
data_label = normalize(data_label, norm='max', axis=0)

label = np.reshape(label, (-1,))

x_test = np.load('data/data_label_test.pkl', allow_pickle=True)
x_test = np.reshape(x_test, (-1, timesteps*n_features))
x_test = normalize(x_test, norm='max', axis=0)
y_test = np.load('data/label_test.pkl', allow_pickle=True)
y_test = np.reshape(y_test, (-1,))


rf_model = RandomForestClassifier().set_params(**params_rf)
rf_model.fit(data_label, label)

print("RF model evaluation on train data: ", categorical_evaluation(rf_model, data_label, label))
print("RF model evaluation on test data: ", categorical_evaluation(rf_model, x_test, y_test))
pickle.dump(rf_model, open("saved_models/model_RF.sav", 'wb'))

svm_model = SVC(probability=True, random_state=37).set_params(**params_svm)
svm_model.fit(data_label, label)

print("SVM model evaluation on train data: ", categorical_evaluation(svm_model, data_label, label))
print("SVM model evaluation on test data: ", categorical_evaluation(svm_model, x_test, y_test))
pickle.dump(svm_model, open("saved_models/model_SVM.sav", 'wb'))

gp_model = GaussianProcessClassifier(random_state=37).set_params(**params_gp)
gp_model.fit(data_label, label)

print("GP model evaluation on train data: ", categorical_evaluation(gp_model, data_label, label))
print("GP model evaluation on test data: ", categorical_evaluation(gp_model, x_test, y_test))
pickle.dump(gp_model, open("saved_models/model_GP.sav", 'wb'))

lstm_model = deepLSTM(timesteps=timesteps, n_features=n_features, **lstm_params)
lstm_model.compile(optimizer=Adam(lr=lstm_params['lr']), loss=lstm_params['loss'], metrics=['acc'])
history = lstm_model.fit(data_label, label_enc, batch_size=32, epochs=epochs, verbose=0)

print("LSTM model evaluation on train data: ", categorical_evaluation(lstm_model, data_label, label_enc))
print("LSTM model evaluation on test data: ", categorical_evaluation(lstm_model, x_test, y_test))
lstm_model.save("saved_models/model_LSTM.h5")

bilstm_model = deepBiLSTM(timesteps=timesteps, n_features=n_features, **bilstm_params)
bilstm_model.compile(optimizer=Adam(lr=bilstm_params['lr']), loss=bilstm_params['loss'], 
                         metrics=['acc'])
history = bilstm_model.fit(data_label, label_enc, batch_size=32, epochs=epochs, verbose=0)

print("BiLSTM model evaluation on train data: ", categorical_evaluation(bilstm_model, data_label, label_enc))
print("BiLSTM model evaluation on test data: ", categorical_evaluation(bilstm_model, x_test, y_test))
lstm_model.save("saved_models/model_BiLSTM.h5")

res_model = ResNet_model(n_feat=n_features, **res_params)
res_model.compile(optimizer=Adam(lr=res_params['lr']), loss=res_params['loss'], metrics=['acc'])
history = res_model.fit(data_label, label_enc, batch_size=32, epochs=epochs, verbose=0)

print("ResNet model evaluation on train data: ", categorical_evaluation(res_model, data_label, label_enc))
print("ResNet model evaluation on test data: ", categorical_evaluation(res_model, x_test, y_test))
res_model.save("saved_models/model_ResNet.h5")

vgg_model = VGG_model(n_feat=n_features, **vgg_params)
vgg_model.compile(optimizer=Adam(lr=vgg_params['lr']), loss=vgg_params['loss'], metrics=['acc'])
history = vgg_model.fit(data_label, label_enc, batch_size=32, epochs=epochs, verbose=0)

print("VGG model evaluation on train data: ", categorical_evaluation(vgg_model, data_label, label_enc))
print("VGG model evaluation on test data: ", categorical_evaluation(vgg_model, x_test, y_test))
vgg_model.save("saved_models/model_VGG.h5")

inc_model = Inception_model(n_feat=n_features, **inc_params)
inc_model.compile(optimizer=Adam(lr=inc_params['lr']), loss=inc_params['loss'], metrics=['acc'])
history = inc_model.fit(data_label, label_enc, batch_size=32, epochs=epochs, verbose=0)

print("Inception model evaluation on train data: ", categorical_evaluation(inc_model, data_label, label_enc))
print("Inception model evaluation on test data: ", categorical_evaluation(inc_model, x_test, y_test))
inc_model.save("saved_models/model_Inception.h5")
