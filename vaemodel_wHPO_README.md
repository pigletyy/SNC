# README File for VAE Model with HPO
This file documents the key funtions and code flow of the **vaemodel_wHPO.ipynb** file.\
**Prepared & last edited by:** Lee Yng-Yng (21/05/2024)

## Libraries Requirements
All the necessary packages and the windows requirements can be found in the .txt document \
titled "lyyrequirements.txt". Should there be any conflict between package/library version(s) \
etc. please refer to the above document to align system requirements.

## Usage
### 1. Run Architectural and Training Loops
The first step would be to run all the cells right up until the "Main Loop & Hyperparameter Tuning" section.
Upon completion, the basic VAE model architecture creation function is initialized, alongside all the training/testing loops functions that will be called on from the main loop.

### 2. Upload Datasets (Main Loop)
Edit the names of the reelevant dataset files as applicable.\
For example, the following code is used to retrieve the features and labels .csv files.\
Replace the file names such as "deepship_standard_train_features.csv" accordingly.
```python
train_features = pd.read_csv("deepship_standard_train_features.csv")
train_labels = pd.read_csv("deepship_standard_train_labels.csv")
```

### 3. Run Data Pre-Processing / Feature Engineering
Run all the remaining cells in the "Main Loop & Hyperparameter Tuning" section.\
The seperate code chunks and their purpose are briefly introduced below:
* sklearn's Variance Threshold\
Feature selector that removes low-variance (less salient) features.\
class sklearn.feature_selection.VarianceThreshold(threshold = 0.9)\
Edit the variance threshold value accordingly. Selected features will then be retained for validation and test feature sets.\
i.e. threshold=0.9: all train features with absolute variance < 0.9 will be dropped
```python
sel  = VarianceThreshold(threshold = (0.9)) # adjust threshold accordingly
train_features = sel.fit_transform(train_features)
selected_features = sel.get_feature_names_out()
eval_features = eval_features[selected_features]
test_features = test_features[selected_features]
```
* Principal Component Analysis (PCA)\
Dimensionality reduction ML method used to simplify large datasets into smaller sets while maintaining significant patterns and trends.\
Used in this project simply to get a rough gauge of how many features are required to represent a certain variance threshold of the dataset.\
Adjust the threshold_index variable in the following code accordingly to get the approximate number of features required.
```python
threshold_index = np.argmax(y >= 0.90) + 1
```
 
* Data Normalization\
Feature scaling technique to transform features to be on a similar scale. The following scalers are used as appropriate:
    * MaxAbsScaler: scales data to range [-1, 1]
    * StandardScaler:
    * MinMaxScaler: type\

    The following code selects the most suitable scaler according to data characteristics and fits it onto the training dataset.
    ```python
    sparsity = (train_features==0).sum().sum() / (train_features.size - train_features.shape[1])
    sparsity_threshold = 0.1
    if sparsity > sparsity_threshold: # More than 30% of data is 0-valued
        scaler = pp.MaxAbsScaler() # MaxAbs Scaler: scale data to range [-1, 1]
    else:
        negatives = (train_features < 0).any().any()
        if negatives:
            scaler = pp.StandardScaler()
        else:
            scaler = pp.MinMaxScaler()
    ```

### 4. Hyperparameter Tuning & Model Selection