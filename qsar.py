
"""QSAR.ipynb



Inspired from the following source
    https://github.com/dataprofessor/bioactivity-prediction-app/blob/main/app.py
# **QSAR Model
"""

import pandas as pd

dataset = pd.read_csv('chembl262_chembl2850_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')
dataset.dropna()
save=dataset.dropna()
dataset.to_csv('chembl262_chembl2850_06_bioactivity_data_3class_pIC50_pubchem_fp_nan.csv')
X = dataset.drop(['pIC50'], axis=1)
Y = dataset.iloc[:,-1]

from sklearn.feature_selection import VarianceThreshold

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X = remove_low_variance(X, threshold=0.1)

X.to_csv('descriptor_list.csv', index = False)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train.shape, Y_train.shape

X_test.shape, Y_test.shape


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#LGBM Regressor
import lightgbm as ltb
model = ltb.LGBMRegressor(random_state=100, n_estimators=100, max_depth=50)
model.fit(X, Y)
r2= model.score(X, Y)
r2

# Histogram gradiant boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=100, n_estimators=800)
model.fit(X_train, Y_train)
r2= model.score(X_test, Y_test)
r2

# Grid Search CV
search_space = {
    "n_estimators" : [100, 200, 500],
    "max_depth" : [3, 6, 9],
    "gamma" : [0.01, 0.1],
    "learning_rate" : [0.001, 0.01, 0.1, 1]
}

#from sklearn.model_selection import GridSearchCV
# make a GridSearchCV object
GS = GridSearchCV(estimator = xgb_model,
                  param_grid = search_space,
                  scoring = ["r2", "neg_root_mean_squared_error"], #sklearn.metrics.SCORERS.keys()
                  refit = "r2",
                  cv = 5,
                  verbose = 4)

GS.fit(X_train, Y_train)

print(GS.best_estimator_) # to get the complete details of the best model

print(GS.best_params_) # to get only the best hyperparameter values that we searched for

print(GS.best_score_) # score according to the metric we passed in refit

df = pd.DataFrame(GS.cv_results_)
df = df.sort_values("rank_test_r2")
df.to_csv("cv_results.csv", index = False)

Y_pred = model.predict(X)
Y_pred
import matplotlib.pyplot as plt
x_ax = range(len(Y_test))
plt.figure(figsize=(20,20))
plt.plot(x_ax, Y_test, label="original")
plt.plot(x_ax, Y_pred, label="predicted")
plt.title("Test and predicted data")
plt.legend()
plt.show()

plt.savefig("Orignal and predicted data in plot.pdf")

# Commented out IPython magic to ensure Python compatibility.
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y, Y_pred))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y, Y_pred))

"""# Data Visualization (Experimental vs Predicted pIC50 for Training Data)"""

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,10))
plt.scatter(Y_test, Y_pred, c="#7CAE00", alpha=0.3)

z = np.polyfit(Y, Y_pred, 1)
#p = np.poly1d(z)

#plt.plot(Y,p(Y),"#F8766D")
#plt.ylabel('Predicted pIC50',fontsize=20)
#plt.xlabel('Experimental pIC50', fontsize=20)

#plt.savefig("Data Visualization (Experimental vs Predicted pIC50 for Training Data.pdf")

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)
sns.set_style("white")

ax = sns.regplot(X=Y, Y=Y_pred, scatter_kws={'alpha':0.4})
ax.set_xlabel('Experimental pIC50', fontsize=20, fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize=20, fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.figure.set_size_inches(10, 10)
plt.show
plt.savefig("Data Visualization (Experimental vs Predicted pIC50 for Training Data.pdf")

import pickle

pickle.dump(model, open('chembl262_chembl2850_model_XGB_regressor_new.pkl', 'wb'))

