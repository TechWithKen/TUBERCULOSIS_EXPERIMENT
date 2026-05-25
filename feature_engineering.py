from sklearn.model_selection import train_test_split
from data_cleaning import apply_label
from dataloading import load_data


def split_features(df):
    target = "tb_risk"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)


    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


def get_feature_groups(X):
    categorical = list(X.select_dtypes(include=['object']).columns)
    numerical = list(X.select_dtypes(exclude=['object']).columns)

    return categorical, numerical
