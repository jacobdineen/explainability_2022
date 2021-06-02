# Perturbation-Explanation

Code Repo for Model Explainability in Predictive Analytics: A Comparative Approach to Unified Explanations

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```python
pip install -r requirements.txt
```

## Usage

### Define a set of algorithms and metrics
```python
models = {
    'nn':
    MLPClassifier(random_state=0),
#     'svm':
#     sklearn.svm.SVC(kernel='rbf', probability=True),
    'logit':
    sklearn.linear_model.LogisticRegression(),
    'rf':
    sklearn.ensemble.RandomForestClassifier(n_estimators=100,
                                            max_depth=None,
                                            min_samples_split=2,
                                            random_state=0),
    'knn':
    sklearn.neighbors.KNeighborsClassifier(),
    'gbc':
    sklearn.ensemble.GradientBoostingClassifier(random_state=0)
}

metrics = {
    'accuracy': lambda Y,Y_hat : accuracy_score(Y,Y_hat), 
    'f1': lambda Y,Y_hat : f1_score(Y,Y_hat,average='weighted', zero_division = 0), 
    'precision': lambda Y,Y_hat : precision_score(Y,Y_hat,average='weighted', zero_division = 0), 
    'recall': lambda Y,Y_hat : recall_score(Y,Y_hat,average='weighted', zero_division = 0), 
}
```




## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)