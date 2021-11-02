# Perturbation-Explanation

Code Repo for Model Explainability in Predictive Analytics: A Comparative Approach to Unified Explanations

## Installation

Run git clone to store the repository locally
```
git clone https://github.com/jacobdineen/explainability_2022.git
``` 

create a conda env and install required modules
'''
conda create --name hicss22 --file requirements.txt
```



## Usage

### Fetching Datasets
By default, we load 4 datasets from the sklearn.datasets API (Iris, Adult, Wine, Cancer). Details on this fetch can be found in the dataloader.py file.
fetchdatasets return a dict of the four datasets, split into train and test sets. fetch_datasets_adult_samples is used to partition varying sizes of the adult dataset for sensitivity analysis. 

The synthetic dataset from JPMC is hidden from this repo.


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

### Run Perturbation Explanation Against Shap

Running the below method will result in two outputs saved to disk if they are not already there. The similarity df shows the cosine sim between the weighted average computed our way, and the output of Shap attribution generation. The granular_value df outputs feature-wise similarity for each of the datasets/models.

```python
from dataloader import fetch_datasets
from attribution import compare_cases

datasets = fetch_datasets()

def run(datasets, metrics, models, load_from_disk = True):
    if load_from_disk:
        similarity_df = pd.read_csv('data/similarity.csv')
        granular_value_df = pd.read_csv('data/gran_value.csv')
        print('loaded from disk')
        return similarity_df, granular_value_df
    else:
        granular_value_df = pd.DataFrame()
        similarity_df = pd.DataFrame()
        for dataset_name, dataset in datasets.items():
            for metric_name, metric_func in metrics.items():
                print(f'dataset: {dataset_name}, metric: {metric_name}')
                logs, vals = compare_cases(dataset=(dataset_name, dataset),
                                           metric=(metric_name, metric_func),
                                           models = models)
                granular_value_df = granular_value_df.append(vals)
                similarity_df = similarity_df.append(logs)
        similarity_df.to_csv('data/similarity.csv')
        granular_value_df.to_csv('data/gran_value.csv')
        print('saved to disk')
        return similarity_df, granular_value_df


similarity_df, granular_value_df = run(datasets, metrics, models, False)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)