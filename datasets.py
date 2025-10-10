import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# List of proper openml versions for each dataset
datasets = {
    "balance-scale": 1,
    "car": 3,
    "cleveland": 1,
    "dermatology": 1,
    "glass": 1,
    "vehicle": 1,
}


def load_dataset(dataset_name):
    dataset = fetch_openml(dataset_name,
                           version=datasets[dataset_name],
                           as_frame=True)
    X = dataset.data
    y = dataset.target

    # Preprocess nominal features and missing values
    categorical_cols = X.select_dtypes(include=['category']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['category']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_cols),  # keep as-is
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    X = preprocessor.fit_transform(X)
    X = X.astype(np.float64)

    # One-hot encoding of labels
    if y.dtype != int:
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.to_numpy()

    return X, y

def create_split(X, y, test_size=0.2, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Apply data scaler without data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
