import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import statsmodels.api as sm

import pickle
import warnings
warnings.filterwarnings("ignore")

# load data
df_dbt = pd.read_csv('./data/diabetes_binary_health_indicators_BRFSS2015.csv')
print(df_dbt.shape)
df_dbt.head(2)


df_dbt['Diabetes_binary'].value_counts(dropna=False, normalize=True)


# # Split Data to Train/Test
df_train, df_test = train_test_split(df_dbt, test_size=0.2, random_state=42)

X = df_dbt.drop('Diabetes_binary', axis=1)
y = df_dbt[['Diabetes_binary']]

X_train = df_train.drop('Diabetes_binary', axis=1)
X_test = df_test.drop('Diabetes_binary', axis=1)

y_train = df_train[['Diabetes_binary']]
y_test = df_test[['Diabetes_binary']]

print(X_train.shape, X_test.shape)

y_test.value_counts(normalize=True)


# # Data Preprocessing
scaler = StandardScaler()

# fir scaler
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled)
X_train_scaled.columns = X_train.columns
X_train_scaled.index = X_train.index
X_train_scaled.head()

# apply to test
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled)
X_test_scaled.columns = X_test.columns
X_test_scaled.index = X_test.index
X_test_scaled.head(2)


# # Modeling
results = pd.DataFrame(columns=['Model', 'Train Score', 'CV Score', 'Test Score', 'Params'])


# ## Logistic Rergession
model_name = 'Logistic Regression'
model_idx = 0

model = LogisticRegression(random_state=42, max_iter=500)

param_grids = {'penalty': [None, 'l2'], 
               'C': [0.01, 0.1, 1, 10]}

grid_search = GridSearchCV(model, param_grids, cv=5, scoring='f1_macro')
grid_search.fit(X_train_scaled, y_train)

# train_score = f1_score(y_train, grid_search.predict(X_train_scaled), average='macro')
cv_score = grid_search.best_score_
# test_score = f1_score(y_test, grid_search.predict(X_test_scaled), average='macro')

# results.loc[model_idx] = [model_name, train_score, cv_score, test_score, grid_search.best_params_]
model_lr = grid_search.best_estimator_
print(model_lr.intercept_)

# find the right threshold for cutting prediction
y_pred = model_lr.predict_proba(X_train_scaled)[:, 1]
list_f1score_lr = []
for i in np.arange(0, 1, 0.01):
    list_f1score_lr.append(f1_score(y_train, y_pred>=i, average='macro'))

ind_lr = np.argmax(list_f1score_lr)

f1_thresh_lr = np.arange(0, 1, 0.01)[ind_lr]
print(f1_thresh_lr)

train_score = f1_score(y_train, model_lr.predict_proba(X_train_scaled)[:, 1]>f1_thresh_lr, average='macro')
test_score = f1_score(y_test, model_lr.predict_proba(X_test_scaled)[:, 1]>f1_thresh_lr, average='macro')
results.loc[model_idx] = [model_name, train_score, cv_score, test_score, grid_search.best_params_]
print(results)

y_pred = model_lr.predict_proba(X_test_scaled)[:, 1]
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred>f1_thresh_lr))
print(classification_report(y_test, y_pred>f1_thresh_lr))

# Create Feature Importances DataFrame and sort
feature_df = pd.DataFrame({'feature': model_lr.feature_names_in_, 'importance': model_lr.coef_[0]})
feature_df = feature_df.sort_values(by='importance', ascending=False)  

# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.barh(feature_df['feature'], feature_df['importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title(f'Logistic Regression Feature Importances')
plt.show()

feature_df['importance_abs'] = abs(feature_df['importance'])
feature_df.sort_values('importance_abs', ascending=False)
print(feature_df)

# ### using SM
logit_model = sm.Logit(y_train, sm.add_constant(X_train_scaled))
result = logit_model.fit()
print(result.summary())


# ## Decision Tree
model_name = 'Decision Tree'
model_idx = 1

model = DecisionTreeClassifier(random_state=42)

param_grids = {'max_depth': [3, 5,
                             7, 9, 11],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'criterion': ['gini', 'entropy']}

grid_search = GridSearchCV(model, param_grids, cv=5, scoring='f1_macro')
grid_search.fit(X_train_scaled, y_train)

# train_score = f1_score(y_train, grid_search.predict(X_train_scaled), average='macro')
cv_score = grid_search.best_score_
# test_score = f1_score(y_test, grid_search.predict(X_test_scaled), average='macro')

# results.loc[model_idx] = [model_name, train_score, cv_score, test_score, grid_search.best_params_]
model_dt = grid_search.best_estimator_

# find the right threshold for cutting prediction
y_pred = model_dt.predict_proba(X_train_scaled)[:, 1]
list_f1score_dt = []
for i in np.arange(0, 1, 0.01):
    
    list_f1score_dt.append(f1_score(y_train, y_pred>=i, average='macro'))

ind_dt = np.argmax(list_f1score_dt)

f1_thresh_dt = np.arange(0, 1, 0.01)[ind_dt]
print(f1_thresh_dt)

train_score = f1_score(y_train, model_dt.predict_proba(X_train_scaled)[:, 1]>f1_thresh_dt, average='macro')
test_score = f1_score(y_test, model_dt.predict_proba(X_test_scaled)[:, 1]>f1_thresh_dt, average='macro')
results.loc[model_idx] = [model_name, train_score, cv_score, test_score, grid_search.best_params_]
print(results)

y_pred = model_dt.predict_proba(X_test_scaled)[:, 1]
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred>f1_thresh_dt))
print(classification_report(y_test, y_pred>f1_thresh_dt))

# Extract feature importances
importances = model_dt.feature_importances_

# Create DataFrame and sort
feature_df = pd.DataFrame({'feature': X_train_scaled.columns, 'importance': importances})
feature_df = feature_df.sort_values(by='importance', ascending=False)  

# Plot the feature importances
plt.figure(figsize=(8, 6)) 
plt.barh(feature_df['feature'], feature_df['importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title(f'Decision Tree Feature Importances')
plt.show()

print(feature_df)


# ## Random Forest
# This cell takes time about 35 minutes.
model_name = 'Random Forest'
model_idx = 2

model = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grids = {'n_estimators': [50, 100, 200,
                                300, 500],
                'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2, 5, 10],
               'max_depth': [3, 5, 7, 9, 11]}

grid_search = GridSearchCV(model, param_grids, cv=5, scoring='f1_macro')
grid_search.fit(X_train_scaled, y_train)

# train_score = f1_score(y_train, grid_search.predict(X_train_scaled), average='macro')
cv_score = grid_search.best_score_
# test_score = f1_score(y_test, grid_search.predict(X_test_scaled), average='macro')

# results.loc[model_idx] = [model_name, train_score, cv_score, test_score, grid_search.best_params_]
model_rf = grid_search.best_estimator_

# find the right threshold for cutting prediction
y_pred = model_rf.predict_proba(X_train_scaled)[:, 1]
list_f1score_rf = []
for i in np.arange(0, 1, 0.01):
    
    list_f1score_rf.append(f1_score(y_train, y_pred>=i, average='macro'))

ind_rf = np.argmax(list_f1score_rf)

f1_thresh_rf = np.arange(0, 1, 0.01)[ind_rf]
print(f1_thresh_rf)

train_score = f1_score(y_train, model_rf.predict_proba(X_train_scaled)[:, 1]>f1_thresh_rf, average='macro')
test_score = f1_score(y_test, model_rf.predict_proba(X_test_scaled)[:, 1]>f1_thresh_rf, average='macro')
results.loc[model_idx] = [model_name, train_score, cv_score, test_score, grid_search.best_params_]
print(results)

y_pred = model_rf.predict_proba(X_test_scaled)[:, 1]
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred>f1_thresh_rf))
print(classification_report(y_test, y_pred>f1_thresh_rf))

# Extract feature importances
importances = model_rf.feature_importances_

# Create DataFrame and sort
feature_df = pd.DataFrame({'feature': X_train_scaled.columns, 'importance': importances})
feature_df = feature_df.sort_values(by='importance', ascending=False)  

# Plot the feature importances
plt.figure(figsize=(8, 6)) 
plt.barh(feature_df['feature'], feature_df['importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title(f'Random Forest Feature Importances')
plt.show()

print(feature_df)

# ## Save models
# save trained models
# model_pkl_file = "./models/data583_2classdiabetes_3models.pkl"  

# with open(model_pkl_file, 'wb') as file:  
#     pickle.dump(model_lr, file)
#     pickle.dump(model_dt, file)
#     pickle.dump(model_rf, file)
#     pickle.dump(scaler, file)

# load trained models
# model_pkl_file = "./models/data583_2classdiabetes_3models.pkl"  

# file = open(model_pkl_file,'rb')
# model_lr = pickle.load(file)
# model_dt = pickle.load(file)
# model_rf = pickle.load(file)
# scaler = pickle.load(file)
# file.close()


# ## All models performance
results = results.sort_values('Test Score', ascending=False)
plt.figure(figsize=(6, 4))
plt.bar(results['Model'], results['Test Score'])
plt.xlabel('Models')
plt.ylabel('F1-Score')
plt.title('Model Performance Comparison (Testing Dataset)')
plt.show()

# CV Score auto-cut the prediction with a threshold equals to 0.5.
results.drop(columns='CV Score')

# My models have the performance of F1-Scores higher than this work in the Kaggle.
# https://www.kaggle.com/code/ohoodalsohaime/diabetes-indicators-classfication-project-part1
