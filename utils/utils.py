import numpy as np
from keras.utils import to_categorical
import keras.backend as K
from model.model import classifier_model, pnn_classifier_model, cnn_classifier_model
from model.BCNNet import make_volunteer_classifiers, fit_volunteer_calssifiers, generate_volunteer_labels, initialise_prior
from model.VB_iteration import VB_iteration
from utils.evaluation import evaluation


def make_selfsupervised_dataset(transformers, x):
    transform_task = np.asarray(range(len(transformers)+1))
    label = to_categorical(transform_task)
    data = []
    labels = []
    max_shape = x.shape[1]*x.shape[2]*x.shape[3]
    for d in x:
        data.append(d.reshape((-1, max_shape, 1)))
        d = d.reshape((1, x.shape[1], x.shape[2], x.shape[3]))
        for trans in transformers:
            pred = trans.predict(d)
            z = np.zeros((pred.shape[0], max_shape-pred.shape[1]), dtype=pred.dtype)
            data.append(np.reshape(np.concatenate((pred,z), axis=1), (-1, max_shape, 1)))
        labels.append(label)
    
    data = np.reshape(np.asarray(data), (-1, max_shape, 1))
    labels = np.reshape(np.asarray(labels), (-1, 11))
    return data, labels


def extract_features(model, x, pnn=False):
    keras_function = K.function([model.input], model.layers[-4].output)
    out = keras_function(x)
    out = np.reshape(out, (x.shape[0], out.shape[2], out.shape[1]))
    if pnn:
        out = np.reshape(out, (x.shape[0], -1))
    return out


def training_process(x_train, y_train, x_train_labelled, y_train_labelled, x_test, y_test, args, model_type, crowd_type=['nb', 'rf', 'svm', 'gp'], crowd_params=None, cl_model=None):
    shape = x_train.shape[1]
    crowdsourced_models = make_volunteer_classifiers(type_list=crowd_type, params=crowd_params)
    crowdsourced_models = fit_volunteer_calssifiers(crowdsourced_models, 
                                                x_train_labelled.reshape((-1, shape)), y_train_labelled)
    crowdsourced_labels = generate_volunteer_labels(crowdsourced_models, x_train.reshape((-1, shape)), 
                                                args['crowdsourced_labelled_train_data_ratio'])

    prior_param_confusion_matrices = initialise_prior(n_classes=args['n_classes'], n_volunteers=args['n_crowd_members'],
                                                                   alpha_diag_prior=args['confusion_matrix_diagonal_prior'])
    variational_param_confusion_matrices = np.copy(prior_param_confusion_matrices)

    initial_nn_output_for_vb_update = np.random.randn(x_train.shape[0], args['n_classes'])

    q_t, variational_param_confusion_matrices, lower_bound = \
        VB_iteration(crowdsourced_labels, initial_nn_output_for_vb_update, variational_param_confusion_matrices,
                     prior_param_confusion_matrices)

    old_lower_bound = lower_bound
    
    if not cl_model:
        if(model_type == 'pnn'):
            cl_model = pnn_classifier_model(input_shape=(x_train.shape[1], x_train.shape[2]), n_class=args['n_classes'], 
                                batch_norm=True)
        elif(model_type == 'lstm'):
            cl_model = classifier_model(input_shape=(x_train.shape[1], x_train.shape[2]), n_class=args['n_classes'], 
                                batch_norm=True, lstm=True, L2=args['L2'])
        elif(model_type == 'dense'):
            cl_model = classifier_model(input_shape=(x_train.shape[1], x_train.shape[2]), n_class=args['n_classes'], 
                                batch_norm=True, L2=args['L2'])
        else:
            cl_model = cnn_classifier_model(input_shape=(x_train.shape[1], x_train.shape[2]), n_class=args['n_classes'], 
                                batch_norm=True)
        
    op = Adam(lr=args['lr'])
    #loss = BinaryCrossentropy(label_smoothing=0.1)
    cl_model.compile(loss=args['loss'], optimizer=op, metrics=['accuracy'])
    
    # based on predictions from the neural network
    nn_training_accuracy = np.zeros((args['n_epoch'],), dtype=np.float64)  
    # based on approximated posterior for true labels
    posterior_estimate_training_accuracy = np.zeros((args['n_epoch'],), dtype=np.float64)  
    # based on predictions from the neural network on test data
    nn_test_accuracy = np.zeros((args['n_epoch'],), dtype=np.float64)
    nn_training_loss = np.zeros(args['n_epoch'], dtype=np.float64)
    test_eval = []
    training_eval = []
    
    
    for epoch in range(args['n_epoch']):
        print(f'epoch {epoch}:')

        # update of parameters of the neural network
        history = cl_model.fit(x_train, q_t, epochs=1, shuffle=True, batch_size=args['batch_size'], verbose=0)
        # update of approximating posterior for the true labels and confusion matrices
        # get current predictions from a neural network
        nn_output_for_vb_update = cl_model.predict(x_train)
        # for numerical stability
        nn_output_for_vb_update = nn_output_for_vb_update - \
            np.tile(np.expand_dims(np.max(nn_output_for_vb_update, axis=1), axis=1), 
                    (1, nn_output_for_vb_update.shape[1]))

        q_t, variational_param_confusion_matrices, lower_bound = \
            VB_iteration(crowdsourced_labels, nn_output_for_vb_update, variational_param_confusion_matrices,
                         prior_param_confusion_matrices)

        # evaluation
        nn_training_accuracy[epoch] = np.mean(np.argmax(nn_output_for_vb_update, axis=1) == y_train)
        print(f'\t nn training accuracy: {nn_training_accuracy[epoch]}')
        nn_training_loss[epoch] = np.asarray(history.history['loss'])
        print(f'\t nn training loss: {nn_training_loss[epoch]}')
        e = evaluation(y_train, nn_output_for_vb_update)
        training_eval.append(e)
        print(f'\t nn training evaluation: {e}')

        posterior_estimate_training_accuracy[epoch] = np.mean(np.argmax(q_t, axis=1) == y_train)
        print(f'\t posterior estimate training accuracy: {posterior_estimate_training_accuracy[epoch]}')

        nn_test_prediction = cl_model.predict(x_test)
        nn_test_accuracy[epoch] = np.mean(np.argmax(nn_test_prediction, axis=1) == y_test)
        print(f'\t nn test accuracy: {nn_test_accuracy[epoch]}')
        e = evaluation(y_test, nn_test_prediction)
        test_eval.append(e)
        print(f'\t nn test evaluation: {e}')

        # check convergence
        if np.abs((lower_bound - old_lower_bound) / old_lower_bound) < args['convergence_threshold']:
            break

        old_lower_bound = lower_bound
        
    return cl_model, crowdsourced_models, epoch,\
           nn_training_loss, nn_training_accuracy, training_eval,\
           posterior_estimate_training_accuracy, nn_test_accuracy, test_eval