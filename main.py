import argparse
import warnings
warnings.simplefilter('error', UserWarning)

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from brenier import BrenierIsotonicRegression
from calibration import MyCalibratedClassifierCV
from calibration import classwise_ECE, conf_ECE, full_ECE
from datasets import datasets, load_dataset, create_split


def evaluate_calibration(y_true, y_prob_uncalib, y_prob_calib, class_names):
    """
    Evaluate calibration performance
    """
    print("\n" + "="*60)
    print("CALIBRATION EVALUATION")
    print("="*60)
    
    # Convert string labels to numeric for sklearn metrics
    le = LabelEncoder()
    y_true_numeric = le.fit_transform(y_true)
    
    # Overall metrics
    try:
        uncalibrated_logloss = log_loss(y_true_numeric, y_prob_uncalib)
        calibrated_logloss = log_loss(y_true_numeric, y_prob_calib)

        uncalibrated_ece = full_ECE(y_prob_uncalib, y_true)
        calibrated_ece = full_ECE(y_prob_calib, y_true)
        
        print(f"\nOverall Log Loss:")
        print(f"  Uncalibrated: {uncalibrated_logloss:.4f}")
        print(f"  Calibrated:   {calibrated_logloss:.4f}")
        print(f"  Improvement:  {uncalibrated_logloss - calibrated_logloss:.4f}")
        
        print(f"\nFull ECE:")
        print(f"  Uncalibrated: {uncalibrated_ece:.4f}")
        print(f"  Calibrated:   {calibrated_ece:.4f}")
        print(f"  Improvement:  {uncalibrated_ece - calibrated_ece:.4f}")
    except Exception as e:
        print(f"Could not calculate log loss: {e}")
    
    # Per-class Brier scores
    print(f"\nPer-class Brier Scores:")
    print(f"{'Class':<10} {'Uncalibrated':<15} {'Calibrated':<15} {'Improvement':<15}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        y_binary = (y_true == class_name).astype(int)
        
        brier_uncalib = brier_score_loss(y_binary, y_prob_uncalib[:, i])
        brier_calib = brier_score_loss(y_binary, y_prob_calib[:, i])
        improvement = brier_uncalib - brier_calib
        
        print(f"{class_name:<10} {brier_uncalib:<15.4f} {brier_calib:<15.4f} {improvement:<15.4f}")

@ignore_warnings(category=ConvergenceWarning)
def visualize(dataset, model, calibrator):
    X, y = load_dataset(dataset)
    n_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = create_split(X, y, test_size=0.2)

    print(f"Dataset: {args.data}")
    print(f"Classifier: {model}")
    print(f"Calibrator: {calibrator}")

    model.fit(X_train, y_train)
    calibrator.fit(model.predict_proba(X_test), y_test)

    import matplotlib.pyplot as plt
    from utils import plot_calibration_map

    _calib_map = calibrator.predict_proba
    if n_classes >= 4:
        # Visualize by setting the rest of the coordinates to zero
        calib_map = lambda x: \
            _calib_map(np.hstack([x, np.zeros((len(x), n_classes-3))]))[:, :3]
    elif n_classes == 3:
        calib_map = _calib_map

    plot_calibration_map(calib_map, name="Brenier IR", resolution=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Visualization mode
    parser.add_argument('--vis', action='store_true', help='run visualization')

    # Common options
    parser.add_argument('-b', '--bins', default=15, type=int, help='number of bins')
    parser.add_argument('--data', type=str, default='balance-scale', help='dataset name')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    base_model = MLPClassifier()
    calibrator = BrenierIsotonicRegression(n_bins=args.bins)

    if args.vis:
        visualize(args.data, base_model, calibrator)

    else:
        X, y = load_dataset(args.data)
        X_train, X_test, y_train, y_test = create_split(
            X, y, test_size=0.2, random_state=args.seed
        )

        base_classifier = base_model
        calibrated_classifier = MyCalibratedClassifierCV(
            clone(base_classifier), calibrator, cv=3,
        )

        print(f"Dataset: {args.data}")
        print(f"Classifier: {base_classifier}")
        print(f"Calibrator: {calibrator}")

        with ignore_warnings(category=ConvergenceWarning):
            calibrated_classifier.fit(X_train, y_train)

        y_prob_uncalib = calibrated_classifier.predict_proba_uncalib(X_test)
        y_prob_calib = calibrated_classifier.predict_proba(X_test)

        evaluate_calibration(y_test, y_prob_uncalib, y_prob_calib, np.unique(y))
