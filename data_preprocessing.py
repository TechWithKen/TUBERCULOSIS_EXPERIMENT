from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def encode_target(y_train):
    encoder = LabelEncoder()

    y_train_enc = encoder.fit_transform(y_train)

    return y_train_enc, encoder


def build_preprocessor(categorical_features, numerical_features):

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor