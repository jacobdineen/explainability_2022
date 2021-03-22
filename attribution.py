import shap
import sklearn
import numpy as np
from utils import split_features_cat_cont, cosine_similarity, shap_anwa_df
import pandas as pd


def fetch_shap_values(model, xtrain, xtest, ytest, k=50):
    # explain all the predictions in the test set
    med = shap.kmeans(xtrain, k)
    explainer = shap.KernelExplainer(model.predict_proba, med)
    shap_values = explainer.shap_values(xtest)
    svs = []
    # extract shap values corresponding to positive class
    ytest = np.array(ytest)
    for i in range(len(ytest)):
        true_label = ytest[i]
        abs_svs_feat = np.abs(shap_values[true_label][i])
        svs.append(abs_svs_feat)
    abs_norm_svs = np.array(np.sum(svs, axis=0) / np.sum(np.sum(svs, axis=0)))
    return abs_norm_svs


def perturb_continuous(data, feature, perturbation):
    return data[feature] * perturbation


def perturb_categorical(data, feature, perturbation):
    """
    if perturbation >= 1: increase total number of feats by x* 2-p
    else: decrease total number by x * 1 - p
    
    """
    idx_0s = data.index[data[feature] == 0].tolist()  # find all 0 indices
    idx_1s = data.index[data[feature] == 1].tolist()  # find all 1 indices
    # activate features until doubled
    if perturbation >= 1:
        p = 2 - perturbation
        idxs = np.random.choice(idx_0s, int(np.ceil(len(idx_1s) * p)), replace=True)
        data[feature].iloc[idxs,] = 1  # change n indices from 0 to 1

    #     # deactivate features until zeroed
    else:
        try:
            p = 1 - perturbation
            idxs = np.random.choice(idx_1s, int(np.ceil(len(idx_1s) * p)), replace=True)
            data[feature].iloc[idxs,] = 0  # change n indices from 0 to 1
        except:
            pass
    return data[feature]


def abs_norm_weighted_avg(
    model, metric: tuple, X_test, Y_test, p_start=0, p_end=200, p_step=10
):

    # apply lambda function from metrics dict of lambda function
    base = metric[1](model.predict(X_test), Y_test)

    # P is the perturbation set. bounded between [0,2]
    P = [(i / 100) for i in range(p_start, p_end + 1) if i % p_step == 0]

    # split all features
    X, X_cont, X_cat = split_features_cat_cont(X_test)

    W = []  # weighted average for each feature
    for x in X:  # loop through each feat
        w = []
        for p in P:  # loop through each perturbation
            delta = 1 - np.abs(1 - p)
            X_test_clone = (
                X_test.copy(deep=True).reset_index().drop(columns="index")
            )  # make a copy each time
            if x in X_cat:
                X_test_clone[x] = perturb_categorical(X_test_clone, x, p)
            else:
                X_test_clone[x] = perturb_continuous(X_test_clone, x, p)

            y_hat = model.predict(X_test_clone)
            measure = metric[1](Y_test, y_hat)
            w.append(delta * measure)

        summed_delta = sum([(1 - np.abs(1 - p)) for p in P])
        # W contains Weighted averages of metric for feature j
        W.append(sum(w) / summed_delta)
    I = []
    for w in W:
        U = np.sum(np.abs(base - w) for w in W)
        u = np.abs(base - w)
        I.append(u / U)
    return I


def compare_cases(dataset: tuple, metric: tuple, models, k: int = 50):
    """
    dataset: (dataset_name, dataset) tuple
    metric: (metric_name, metric) tuple
    
    Returns
    """
    values_df = pd.DataFrame()  # store metadata on SVs on ANWA
    logs = []

    X_train, X_test, Y_train, Y_test = dataset[1]  # Init Dataset
    for ind, i in enumerate(models.keys()):
        model = models[i]  # get model in models
        model.fit(X_train, Y_train)  # fit model_i

        # Comparison
        # ---------------------------------#
        abs_norm_svs = fetch_shap_values(
            model, X_train, X_test, Y_test, k=k
        )  # generate shap values on model_i

        abs_norm_weighted = abs_norm_weighted_avg(
            model=model, metric=metric, X_test=X_test, Y_test=Y_test
        )  # gen ANWA on model_i

        distance = cosine_similarity(abs_norm_svs, abs_norm_weighted)
        #         print(abs_norm_svs, abs_norm_weighted)
        #         print('dist:', distance)
        # ---------------------------------#
        # Logs
        logs.append((dataset[0], i, metric[0], distance))

        # append (feature, model, metric, shap, anwa) to running df log
        feature_vals = (abs_norm_svs, abs_norm_weighted)
        values_df = values_df.append(
            shap_anwa_df(i, dataset[0], metric[0], feature_vals, X_test)
        )

    return pd.DataFrame(logs), values_df
