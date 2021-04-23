import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# types - 'nb':naive bayes, 'rf':random forest, 'knn':k-nearest neighbor, 'gp':gaussian process, 'svm': support vector machine
# all the classifiers are default
classifier_types = {
    'rf': RandomForestClassifier(),
    'nb': GaussianNB(),
    'svm': SVC(probability=True),
    'gp': GaussianProcessClassifier(),
    'knn': KNeighborsClassifier()
}


def make_volunteer_classifiers(n_volunteers=4, type_list=['nb', 'rf', 'svm', 'gp'], params=None):
    models = []
    if params:
        for (typ, param) in zip(type_list, params):
            models.append(classifier_types[typ].set_params(**param))
    else:
        for typ in type_list:
            models.append(classifier_types[typ])
    return models 


# use part of labelled data to train the models
def fit_volunteer_calssifiers(models, x_train, y_train):
    for model in models:
        model = model.fit(x_train, y_train)
    return models


def generate_volunteer_labels(models, x, labelled_ratio):
    n_volunteers = len(models)
    n_tasks = x.shape[0]
    labels = np.empty(shape=(n_tasks, n_volunteers))
    labels.fill(-1)
    ratio = int(x.shape[0]*labelled_ratio)
    x_ratio = x[0:ratio]
    for volunteer, model in enumerate(models):
        labels[0:ratio, volunteer] = model.predict(x_ratio)
    # vanish unfilled tasks
    return labels


def initialise_prior(n_classes, n_volunteers, alpha_diag_prior):
    """
    Create confusion matrix prior for every volunteer - the same prior for each volunteer
    :param n_classes: number of classes (int)
    :param n_volunteers: number of crowd members (int)
    :param alpha_diag_prior: prior for confusion matrices is assuming reasonable crowd members with weak dominance of a
    diagonal elements of confusion matrices, i.e. prior for a confusion matrix is a matrix of all ones where
    alpha_diag_prior is added to diagonal elements (float)
    :return: numpy nd-array of the size (n_classes, n_classes, n_volunteers)
    """
    alpha_volunteer_template = np.ones((n_classes, n_classes), dtype=np.float64) + alpha_diag_prior * np.eye(n_classes)
    return np.tile(np.expand_dims(alpha_volunteer_template, axis=2), (1, 1, n_volunteers))