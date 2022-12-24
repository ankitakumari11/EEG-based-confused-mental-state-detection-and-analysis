from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
from sklearn.model_selection import KFold
training_data = pd.read_csv("final_data.csv")

training_data["Gender"] = training_data["Gender"].apply(
    lambda x: 1 if x == 'M' else 0)

input_cols = ['Age', 'Gender', 'Ethnicity',
              'Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
target_col = 'Label'


inputs = training_data[input_cols].copy()
targets = training_data[target_col].copy()

numeric_cols = ['Attention', 'Mediation', 'Raw', 'Delta', 'Theta',
                'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
categorical_cols = ['Age', 'Gender', 'Ethnicity']

scaler = MinMaxScaler().fit(inputs[numeric_cols])
inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])


encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(
    inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])
X = inputs[numeric_cols + encoded_cols]

# data = X+target_col
targets.to_csv('Target.csv')
X.to_csv('Inputs.csv')
X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    X, targets, test_size=0.2, random_state=123)
X_train, X_val, train_targets, val_targets = train_test_split(
    X_train_val, Y_train_val, test_size=0.25, random_state=123)


# DECISION TREE

# K-fold average:


kfold = KFold(n_splits=5)
models = []

for train_idxs, val_idxs in kfold.split(X):
    X_train, train_targets = X.iloc[train_idxs], targets.iloc[train_idxs]
    X_val, val_targets = X.iloc[val_idxs], targets.iloc[val_idxs]
    model = DecisionTreeClassifier(
        max_leaf_nodes=500, max_depth=11, random_state=42)
    model.fit(X_train, train_targets)
    models.append(model)
print("run")

pickle.dump(models, open('model.pkl', 'wb'))
