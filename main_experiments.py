import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import Model
from keras.utils import to_categorical
from utils.utils import extract_features, training_process
from utils.evaluation import evaluation
from sklearn.gaussian_process.kernels import Matern

nb_params = {
    'var_smoothing': 0.0001519911082952933
}

knn_params = {'algorithm': 'auto',
              'leaf_size': 1,
              'n_neighbors': 1,
              'p': 2,
              'weights': 'uniform'
}

svm_params = {'C': 100,
              'class_weight': 'balanced',
              'degree': 1,
              'gamma': 1e-06,
              'kernel': 'rbf'
}

gp_params = {
    'kernel': 1**2 * Matern(length_scale=1, nu=1.5)
}

rf_params = {
    'bootstrap': False,
    'max_depth': 80,
    'max_features': 'sqrt',
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'n_estimators': 122
}

crowd_params = [nb_params, rf_params, svm_params, knn_params]
crowd_type = ['nb', 'rf', 'svm', 'knn']

args = {
    'n_classes': 2,
    'crowdsourced_labelled_train_data_ratio': 0.5,   # ratio of train data labelled by crowd members
    'n_crowd_members': 4,
    'confusion_matrix_diagonal_prior': 1e-1,
    'n_epoch': 100,
    'batch_size': 32,
    'convergence_threshold': 1e-6,   # convergence is measured as change in ELBO  
    'lr': 1e-4,  
    'loss': 'binary_crossentropy',   # for pnn change the loss to mse
    'L2': 0.01
}

rseed = 1000
np.random.seed(rseed)
tf.set_random_seed(rseed)

data_label =  np.load('data/data_label_train.pkl', allow_pickle=True)
data_label = np.reshape(data_label, (-1, 8*8*3, 1))
label = np.load('data/label_train.pkl', allow_pickle=True)
label_enc = to_categorical(label)

x_test = np.load('data/data_label_test.pkl', allow_pickle=True)
x_test = np.reshape(x_test, (-1, 8*8*3, 1))
y_test = np.load('data/label_test.pkl', allow_pickle=True)

ss_model = load_model('saved_models/ss_model.hdf5')

x_train, x_train_labelled,\
y_train, y_train_labelled = train_test_split(data_label, label, 
                                             test_size=args['crowdsourced_labelled_train_data_ratio'], 
                                             random_state=13, shuffle=True, stratify=label)

config = cl_models[best_id].get_config()
cl_model_new = Model.from_config(config)

x_feat_train = extract_features(ss_model, x_train)
x_feat_train_labelled = extract_features(ss_model, x_train_labelled)
x_feat_test = extract_features(ss_model, x_test)

cl_model_new, crowd_model, epoch, nn_training_loss,\
nn_training_accuracy, training_eval, posterior_estimate_training_accuracy,\
nn_test_accuracy, test_eval = training_process(x_feat_train, y_train, 
                                               x_feat_train_labelled, 
                                               y_train_labelled, x_feat_test, 
                                               y_test, args, model_type='lstm',
                                               crowd_type=crowd_type,
                                               crowd_params=crowd_params, cl_model=cl_model_new)

training_recall = np.zeros((epoch,), dtype=np.float64)
test_recall = np.zeros((epoch,), dtype=np.float64)
training_prec_rec = np.zeros((epoch,), dtype=np.float64)
test_prec_rec = np.zeros((epoch,), dtype=np.float64)
for idx in range(epoch):
    training_recall[idx] = training_eval[idx]['Recall']
    test_recall[idx] = test_eval[idx]['Recall']
    training_prec_rec[idx] = training_eval[idx]['prec-rec']
    test_prec_rec[idx] = test_eval[idx]['prec-rec']
    
plt.plot(range(epoch), training_recall, label='train recall')
plt.plot(range(epoch), test_recall, label='test recall')
plt.legend()
plt.show()

plt.plot(range(epoch), training_prec_rec, label='train prec-rec')
plt.plot(range(epoch), test_prec_rec, label='test prec-rec')
plt.legend()
plt.show()

y_pred = cl_model_new.predict(x_feat_test)
print("Model test evaluation: ", evaluation(y_test, y_pred))

cl_model_new.save("saved_models/proposed_model.h5")