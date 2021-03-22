import pandas as pd
from scipy import spatial


def split_features_cat_cont(data):
    """Returns split of column names seperated by dtype"""
    X = data.columns
    X_cont = data.select_dtypes(include=["float", "float32", "float64"]).columns
    X_cat = data.select_dtypes(exclude=["float", "float32", "float64"]).columns
    return X, X_cont, X_cat


def vectorize_categorical(X):
    numeric_subset = X.select_dtypes(include=["float", "float32", "float64"])
    categorical_subset = X.select_dtypes(exclude=["float", "float32", "float64"])
    categorical_subset = pd.get_dummies(categorical_subset.astype("str"))
    X = numeric_subset.join(categorical_subset)
    return X


def cosine_similarity(i, j):
    return 1 - spatial.distance.cosine(i, j)


def shap_anwa_df(model_name, data_name, metric_name, feature_vals, X_test):
    df = pd.DataFrame(feature_vals).T
    df = df.reset_index()
    df.columns = ["feature", "shap", "anwa"]
    df["dataset"] = data_name
    df["feature"] = X_test.columns
    df["model"] = model_name
    df["metric"] = metric_name
    return df
