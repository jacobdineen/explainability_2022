import shap
import sklearn
from utils import vectorize_categorical
from sklearn.model_selection import train_test_split


def fetch_datasets():
    """Method to systematically fetch partitioned sets of a
    numerous Shap/Sklearn datasets and store them in a dict
    """

    calls = {
        "iris": shap.datasets.iris(),
        "adult": shap.datasets.adult(),
        "wine": sklearn.datasets.load_wine(return_X_y=True, as_frame=True),
        "cancer": sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True),
    }

    datasets = {}
    for k, v in calls.items():
        if k == "adult":
            X = v[0][:150]
            Y = v[1][:150]
            X = X.iloc[:, :-1]  # drop country column
            X = vectorize_categorical(X)
            v = (X, Y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            *v, test_size=0.2, shuffle=True, random_state=0
        )
        datasets[k] = X_train, X_test, Y_train, Y_test

    return datasets


def fetch_datasets_adult_samples():
    """Method to systematically fetch partitioned sets of a
    numerous Shap/Sklearn datasets and store them in a dict
    """

    calls = {
        "adult_1000": shap.datasets.adult(),
        "adult_2000": shap.datasets.adult(),
        "adult_4000": shap.datasets.adult(),
        "adult_8000": shap.datasets.adult(),
        "adult_16000": shap.datasets.adult(),
        "adult_32000": shap.datasets.adult(),
    }

    datasets = {}
    samples = 1000
    for k, v in calls.items():
        X = v[0][:samples]
        Y = v[1][:samples]
        X = X.iloc[:, :-1]  # drop country column
        X = vectorize_categorical(X)
        v = (X, Y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            *v, test_size=0.2, shuffle=True, random_state=0
        )
        datasets[k] = X_train, X_test, Y_train, Y_test
        samples *= 2

    return datasets
