import numpy as np
import keras
from sklearn import metrics
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# one is agitation and zero is not-agitation
# here positive is agitation and negative is not-agitation
def decision(y_pred, thresh=0.5):
    return np.where(y_pred[:,1] >= thresh, 1, 0)
    #return np.argmax(y_pred, axis=1)


# number of positive cases which have been predicted negative / number of positive cases
def false_rejection_rate(y_true, y_pred, thresh=0.5):
    pred = decision(y_pred, thresh)
    num_p = np.count_nonzero(y_true == 1.)
    FN = [i for i in range(0, len(pred)) if y_true[i] == 1. and pred[i] == 0.]
    num_f_n = len(FN)
    return num_f_n/num_p


# 1 - (number of negative cases which predicted negative / number of negative cases)
def false_acceptance_rate(y_true, y_pred, thresh=0.5):
    pred = decision(y_pred, thresh)
    num_n = np.count_nonzero(y_true == 0.)
    TN = [i for i in range(0, len(pred)) if pred[i] == 0. and y_true[i] == 0.]
    num_t_n = len(TN)
    return 1-(num_t_n/num_n)


def HalfTotalErrorRate(y_true, y_pred, thresh=0.5):
    alpha = 0.5
    if y_true.ndim != 1:
        y_true = np.argmax(y_true, axis=1)
    far = false_acceptance_rate(y_true, y_pred, thresh)
    frr = false_rejection_rate(y_true, y_pred, thresh)
    return (alpha * far) + (alpha * frr)


def CategoricalTruePositive(y_true, y_pred, thresh=0.5):
    y_true = np.argmax(y_true, axis=1)
    y_pred = decision(y_pred, thresh)

    true_poss = np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1))
    true_poss = np.sum(true_poss)
    return true_poss


def CategoricalTrueNegative(y_true, y_pred, thresh=0.5):
    y_true = np.argmax(y_true, axis=1)
    y_pred = decision(y_pred, thresh)
    
    true_neg = np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0))
    true_neg = np.sum(true_neg)
    return true_neg
 

def CategoricalFalseNegative(y_true, y_pred, thresh=0.5):
    y_true = np.argmax(y_true, axis=1)
    y_pred = decision(y_pred, thresh)

    false_neg = np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0))
    false_neg = np.sum(false_neg)
    return false_neg


def CategoricalFalsePositive(y_true, y_pred, thresh=0.5):
    y_true = np.argmax(y_true, axis=1)
    y_pred = decision(y_pred, thresh)

    false_poss = np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1))
    false_poss = np.sum(false_poss)
    return false_poss


class CategoricalEvaluation(keras.callbacks.Callback):
    def __init__(self, model, validation_data, thresh=0.5):
        super().__init__()
        self.model = model
        self.y_true = validation_data[1]
        self.x_val = validation_data[0]
        self.thresh = thresh
        self.CTP = []
        self.CTN = []
        self.CFN = []
        self.CFP = []
        self.HTER = []
        self.recall = []
        self.specificity = []
        self.precision = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_val)
        cat_true_positive = CategoricalTruePositive(self.y_true, y_pred, self.thresh)
        cat_true_negative = CategoricalTrueNegative(self.y_true, y_pred, self.thresh)
        cat_false_negative = CategoricalFalseNegative(self.y_true, y_pred, self.thresh)
        cat_false_positive = CategoricalFalsePositive(self.y_true, y_pred, self.thresh)
        half_total_error_rate = HalfTotalErrorRate(self.y_true, y_pred, self.thresh)
        cat_recall = cat_true_positive / (cat_true_positive + cat_false_negative)
        cat_spec = cat_true_negative / (cat_true_negative + cat_false_positive)
        cat_prec = cat_true_positive / (cat_true_positive + cat_false_positive)
        
        self.CTP.append(cat_true_positive)
        self.CTN.append(cat_true_negative)
        self.CFN.append(cat_false_negative)
        self.CFP.append(cat_false_positive)
        self.HTER.append(half_total_error_rate)
        self.recall.append(cat_recall)
        self.specificity.append(cat_spec)
        self.precision.append(cat_prec)
        
        print("True Positive:", cat_true_positive)
        print("True Negative:", cat_true_negative)
        print("False Negative:", cat_false_negative)
        print("False Positive:", cat_false_positive)
        print("Recall/Sensitivity:", cat_recall)
        print("Specificity:", cat_spec)
        print("Precision:", cat_prec)
        print("Half Total Error Rate:", half_total_error_rate)


def CategoricalAccuracy(thresh=0.5):
    def cat_accuracy(y_true, y_pred):
        y_pred = tf.cast(tf.greater_equal(y_pred[:, 1], thresh), tf.float32)
        y_true = tf.cast(tf.argmax(y_true, axis=1), tf.float32)
        return K.mean(K.equal(y_true, K.round(y_pred)))
    
    return cat_accuracy

def evaluation(y_true, y_pred):
    true_positive = CategoricalTruePositive(y_true, y_pred)
    true_negative = CategoricalTrueNegative(y_true, y_pred)
    false_negative = CategoricalFalseNegative(y_true, y_pred)
    false_positive = CategoricalFalsePositive(y_true, y_pred)
    recall = true_positive / (true_positive + false_negative)
    spec = true_negative / (true_negative + false_positive)
    prec = true_positive / (true_positive + false_positive)
    roc_auc = roc_auc_score(y_true, y_pred[:, 1])
    precision, rec, _ = precision_recall_curve(y_true, y_pred[:, 1])
    pr_rec = auc(rec, precision)
    f1 = 2 * (recall * prec) / (prec + recall)
    acc = (true_positive + true_negative)/(true_negative + true_positive + false_negative + false_positive)

    return {'TP': true_positive,
            'TN': true_negative,
            'FN': false_negative,
            'FP': false_positive,
            'Accuracy': acc,
            'Recall': recall,
            'Specificity': spec,
            'Precision': prec,
            'F1': f1,
            'AUC': roc_auc,
            'prec-rec': pr_rec
           }


def categorical_evaluation(model, x, true, thresh=0.5):
    y_pred = model.predict_proba(x)
    cat_true_positive = CategoricalTruePositive(true, y_pred, thresh)
    cat_true_negative = CategoricalTrueNegative(true, y_pred, thresh)
    cat_false_negative = CategoricalFalseNegative(true, y_pred, thresh)
    cat_false_positive = CategoricalFalsePositive(true, y_pred, thresh)
    half_total_error_rate = HalfTotalErrorRate(true, y_pred, thresh)
    cat_recall = cat_true_positive / (cat_true_positive + cat_false_negative)
    cat_spec = cat_true_negative / (cat_true_negative + cat_false_positive)
    cat_prec = cat_true_positive / (cat_true_positive + cat_false_positive)
    auc_roc = roc_auc_score(true, y_pred[:, 1])
    precision, recall, thresholds = precision_recall_curve(true, y_pred[:, 1])
    pr_rec = auc(recall, precision)
    acc = (cat_true_positive + cat_true_negative)/(cat_true_negative + cat_true_positive + cat_false_negative + cat_false_positive)
    f1 = 2 * (cat_recall * cat_prec)/(cat_recall + cat_prec)
    return {'TP': cat_true_positive,
            'TN': cat_true_negative,
            'FN': cat_false_negative,
            'FP': cat_false_positive,
            'Accuracy': acc,
            'Recall': cat_recall,
            'Specificity': cat_spec,
            'Precision': cat_prec,
            'F1': f1,
            'HTER': half_total_error_rate,
            'auc': auc_roc,
            'prec-rec': pr_rec
           }