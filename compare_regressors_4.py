
"""Compare_Regressors.ipynb
Inspired from the following source
https://github.com/dataprofessor/code/blob/master/python/CDD_ML_Part_5_Acetylcholinesterase_Compare_Regressors.ipynb

"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyRegressor

! wget https://github.com/dataprofessor/data/raw/master/acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv

df = pd.read_csv('chembl262_chembl2850_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')

X = df.drop('pIC50', axis=1)
Y = df.pIC50

from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
X = selection.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, Y_train, Y_test)
save_1= models
save_1.to_csv('Performance of table training set(80%).csv')
save_2= predictions
save_2.to_csv('Performance of table test set(20%).csv')
save_2

import matplotlib.pyplot as plt
import seaborn as sns

train["R-Squared"] = [0 if i < 0 else i for i in train.iloc[:,0] ]

plt.figure(figsize=(5, 20))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=models.index, x="R-Squared", data=models)
ax.set(xlim=(0, 1))

plt.savefig('data visualization of model performance.pdf')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=models.index, x="RMSE", data=models)
ax.set(xlim=(0, 10))

plt.savefig('Bar Plot of RMSE values.pdf')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=models.index, x="Time Taken", data=models)
ax.set(xlim=(0, 10))

plt.savefig('Bar Plot of calculation time.pdf')
