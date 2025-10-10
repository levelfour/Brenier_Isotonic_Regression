import numpy as np
from numpy.testing import assert_array_equal
from scipy.special import softmax

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


# Implemented with help of claude
class MyCalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    """
    A custom implementation of calibrated classifier with cross-validation.
    
    This class takes an arbitrary base model and applies probability calibration
    using cross-validation, similar to sklearn's CalibratedClassifierCV.
    
    Parameters
    ----------
    base_estimator : estimator instance
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs.
        
    calibration_method : calibrator instance
        The calibration method to use. Must have fit(decision_scores, y) and
        predict_proba(decision_scores) methods.
        
    cv : int, cross-validation generator or iterable, default=3
        Determines the cross-validation splitting strategy.
        If int, to specify the number of folds in StratifiedKFold.
    """
    
    def __init__(self, base_estimator, calibration_method, cv=3, random_state=None):
        self.base_estimator = base_estimator
        self.calibration_method = calibration_method
        self.cv = cv
        self.random_state = random_state
        self.calibrators = []
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the calibrated model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently not supported.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if sample_weight is not None:
            warnings.warn("sample_weight is not supported yet")
            
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr'])
        check_classification_targets(y)
        
        # Store classes and encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        self._fit(X, y_encoded)

    def _fit(self, X, y):
        clf = clone(self.base_estimator)
        clf.fit(X, y)

        if hasattr(clf, "decision_function"):
            scores = cross_val_predict(clf, X, y, cv=self.cv, method="decision_function")
        elif hasattr(clf, "predict_proba"):
            scores = cross_val_predict(clf, X, y, cv=self.cv, method="predict_proba")
        else:
            raise ValueError("Base estimator must have either decision_function or predict_proba")

        calibrator = clone(self.calibration_method)
        calibrator.fit(scores, y)
        self.base_estimator_ = clf
        self.calibrators.append(calibrator)
    
    def predict_proba(self, X):
        """
        Predict calibrated probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples.
            
        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            The predicted probabilities for each class.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=['csc', 'csr'])

        # Get decision scores from fitted base estimator
        if hasattr(self.base_estimator_, "decision_function"):
            decision_scores = self.base_estimator_.decision_function(X)
        elif hasattr(self.base_estimator_, "predict_proba"):
            decision_scores = self.base_estimator_.predict_proba(X)
        else:
            raise ValueError("Base estimator must have either decision_function or predict_proba")

        # Predict averaged probability over calibrators trained on cv splits
        calibrated_probas = [c.predict_proba(decision_scores) for c in self.calibrators]
        calibrated_probas = np.array(calibrated_probas)
        return calibrated_probas.mean(axis=0)

    def predict_proba_uncalib(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=['csc', 'csr'])

        # Get decision scores from fitted base estimator
        if hasattr(self.base_estimator_, "decision_function"):
            decision_scores = self.base_estimator_.decision_function(X)
            return softmax(decision_scores, axis=-1)
        elif hasattr(self.base_estimator_, "predict_proba"):
            return self.base_estimator_.predict_proba(X)
        else:
            raise ValueError("Base estimator must have either decision_function or predict_proba")

    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        proba = self.predict_proba(X)
        predictions_encoded = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(predictions_encoded)



"""
Ported from https://github.com/dirichletcal/experiments_neurips/blob/54989edf11995f7f0f8390766e51133373044db2/calib/utils/functions.py
"""

def binary_ECE(probs, y_true, power = 1, bins = 15):

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1
    bin_func = lambda p, y, idx: (np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power) * np.sum(idx) / len(probs)

    ece = 0
    for i in np.unique(idx):
        ece += bin_func(probs, y_true, idx == i)
    return ece

def classwise_ECE(probs, y_true, power = 1, bins = 15):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    n_classes = probs.shape[1]

    return np.mean(
        [
            binary_ECE(
                probs[:, c], y_true[:, c].astype(float), power = power, bins = bins
            ) for c in range(n_classes)
        ]
    )

def conf_ECE(probs, y_true, bins=15):
    """
    Calculate ECE score based on model output probabilities and true labels

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: - a list containing the actual class labels
                - ndarray shape (n_samples) with a list containing actual class
                labels
                - ndarray shape (n_samples, n_classes) with largest value in
                each row for the correct column class.
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        ece - expected calibration error
    """
    return ECE(probs, y_true, normalize=False, bins=bins, ece_full=False)

def ECE(probs, y_true, normalize = False, bins = 15, ece_full = True):
    """
    Calculate ECE score based on model output probabilities and true labels

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: - a list containing the actual class labels
                - ndarray shape (n_samples) with a list containing actual class
                labels
                - ndarray shape (n_samples, n_classes) with largest value in
                each row for the correct column class.
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized. (default = False)
        bins: (int) - into how many bins are probabilities divided (default = 15)
        ece_full: (bool) - whether to use ECE-full or ECE-max.

    Returns:
        ece - expected calibration error
    """

    probs = np.array(probs)
    y_true = np.array(y_true)
    if len(y_true.shape) == 2 and y_true.shape[1] > 1:
        y_true = y_true.argmax(axis=1).reshape(-1, 1)

    # Prepare predictions, confidences and true labels for ECE calculation
    if ece_full:
        preds, confs, y_true = get_preds_all(probs, y_true, normalize=normalize, flatten=True)

    else:
        preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction

        if normalize:
            confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            confs = np.max(probs, axis=1)  # Take only maximum confidence


    # Calculate ECE and ECE2
    ece = ECE_helper(confs, preds, y_true, bin_size = 1/bins, ece_full = ece_full)

    return ece


def get_preds_all(y_probs, y_true, axis = 1, normalize = False, flatten = True):
    """
    Method to get predictions in right format for ECE-full.

    Params:
        y_probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        axis: (int) dimension of set to calculate probabilities on
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized. (default = False)
        flatten: (bool) - flatten all the arrays

    Returns:
        (y_preds, y_probs, y_true) - predictions, probabilities and true labels
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    elif len(y_true.shape) == 2 and y_true.shape[1] > 1:
        y_true = y_true.argmax(axis=1).reshape(-1, 1)

    y_preds = np.argmax(y_probs, axis=axis)  # Take maximum confidence as prediction
    y_preds = y_preds.reshape(-1, 1)

    if normalize:
        y_probs /= np.sum(y_probs, axis=axis)

    n_classes = y_probs.shape[1]
    y_preds = label_binarize(y_preds, classes=range(n_classes))
    y_true = label_binarize(y_true, classes=range(n_classes))

    if flatten:
        y_preds = y_preds.flatten()
        y_true = y_true.flatten()
        y_probs = y_probs.flatten()

    return y_preds, y_probs, y_true


def ECE_helper(conf, pred, true, bin_size = 0.1, ece_full = False):

    """
    Expected Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        ece: expected calibration error
    """

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true, ece_full)
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE

    return ece


def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true,
                    ece_full=True):
    """
    # Computes accuracy and average confidence for bin

    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        pred_thresh (float) : float in range (0,1), indicating the prediction threshold

    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if  (x[2] > conf_thresh_lower or conf_thresh_lower == 0)  and x[2] <= conf_thresh_upper]

    if len(filtered_tuples) < 1:
        return 0.0, 0.0, 0
    else:
        if ece_full:
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples])/len_bin  # Avg confidence of BIN
            accuracy = np.mean([x[1] for x in filtered_tuples])  # Mean difference from actual class
        else:
            correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
            accuracy = float(correct)/len_bin  # accuracy of BIN

    return accuracy, avg_conf, len_bin

def full_ECE(probs, y_true, bins = 15, power = 1):
    n = len(probs)

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1

    filled_bins = np.unique(idx, axis=0)

    s = 0
    for bin in filled_bins:
        i = np.where((idx == bin).all(axis=1))[0]
        s += (len(i)/n) * (
            np.abs(np.mean(probs[i], axis=0) - np.mean(y_true[i], axis=0))**power
        ).sum()

    return s
