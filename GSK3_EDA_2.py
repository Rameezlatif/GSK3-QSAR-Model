# -*- coding: utf-8 -*-
"""Exploratory_Data_Analysis.ipynb
Inspired from the following source

Inspired from the following source
    https://github.com/dataprofessor/code/blob/master/python/CDD_ML_Part_1_Bioactivity_Data_Concised.ipynb
"""

! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
! chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
! bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local
! conda install -c rdkit rdkit -y
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')

import pandas as pd

df = pd.read_csv('chembl262_chembl2850_03_bioactivity_data_curated.csv')
df

df_no_smiles = df.drop(columns='canonical_smiles')

smiles = []

for i in df.canonical_smiles.tolist():
  cpd = str(i).split('.')
  cpd_longest = max(cpd, key = len)
  smiles.append(cpd_longest)

smiles = pd.Series(smiles, name = 'canonical_smiles')

df_clean_smiles = pd.concat([df_no_smiles,smiles], axis=1)
df_clean_smiles

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# Inspired by: https://codeocean.com/explore/capsules?query=tag:data-curation

def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors

df_lipinski = lipinski(df_clean_smiles.canonical_smiles)
df_lipinski

df_lipinski

df

df_combined = pd.concat([df,df_lipinski], axis=1)

df_combined

# https://github.com/chaninlab/estrogen-receptor-alpha-qsar/blob/master/02_ER_alpha_RO5.ipynb

import numpy as np

def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)
        
    return x

"""Point to note: Values greater than 100,000,000 will be fixed at 100,000,000 otherwise the negative logarithmic value will become negative."""

df_combined.standard_value.describe()

-np.log10( (10**-9)* 100000000 )

-np.log10( (10**-9)* 10000000000 )

def norm_value(input):
    norm = []

    for i in input['standard_value']:
        if i > 100000000:
          i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', 1)
        
    return x

df_norm = norm_value(df_combined)
df_norm

df_norm.standard_value_norm.describe()

df_final = pIC50(df_norm)
df_final

df_final.pIC50.describe()

"""Let's write this to CSV file."""

df_final.to_csv('chembl262_chembl2850_04_bioactivity_data_3class_pIC50.csv')

df_2class = df_final[df_final['class'] != 'intermediate']
df_2class

df_2class.to_csv('chembl262_chembl2850_05_bioactivity_data_2class_pIC50.csv')

"""---"""

import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt

plt.figure(figsize=(10.5, 10.5))

sns.countplot(x='class', data=df_2class, edgecolor='black')

plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('Frequency', fontsize=20, fontweight='bold')

plt.savefig('plot_bioactivity_class.pdf')

plt.figure(figsize=(10.5, 10.5))

sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='class', size='pIC50', edgecolor='black', alpha=0.7)

plt.xlabel('MW', fontsize=20, fontweight='bold')
plt.ylabel('LogP', fontsize=20, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig('plot_MW_vs_LogP.pdf')

plt.figure(figsize=(10.5, 10.5))

sns.boxplot(x = 'class', y = 'pIC50', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=20, fontweight='bold')

plt.savefig('plot_ic50.pdf')

def mannwhitney(descriptor, verbose=False):
  # https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
  from numpy.random import seed
  from numpy.random import randn
  from scipy.stats import mannwhitneyu

# seed the random number generator
  seed(1)

# actives and inactives
  selection = [descriptor, 'class']
  df = df_2class[selection]
  active = df[df['class'] == 'active']
  active = active[descriptor]

  selection = [descriptor, 'class']
  df = df_2class[selection]
  inactive = df[df['class'] == 'inactive']
  inactive = inactive[descriptor]

# compare samples
  stat, p = mannwhitneyu(active, inactive)
  #print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
  alpha = 0.05
  if p > alpha:
    interpretation = 'Same distribution (fail to reject H0)'
  else:
    interpretation = 'Different distribution (reject H0)'
  
  results = pd.DataFrame({'Descriptor':descriptor,
                          'Statistics':stat,
                          'p':p,
                          'alpha':alpha,
                          'Interpretation':interpretation}, index=[0])
  filename = 'mannwhitneyu_' + descriptor + '.csv'
  results.to_csv(filename)

  return results

save= mannwhitney('pIC50')
save
save.to_csv('Mann-Whitney U Test_pIC50.csv')

plt.figure(figsize=(10.5, 10.5))

sns.boxplot(x = 'class', y = 'MW', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('MW', fontsize=20, fontweight='bold')

plt.savefig('plot_MW.pdf')

save_2= mannwhitney('MW')
save_2.to_csv('Mann-Whitney U Test_pIC50_MW.csv')
save_2

plt.figure(figsize=(10.5, 10.5))

sns.boxplot(x = 'class', y = 'LogP', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('LogP', fontsize=20, fontweight='bold')

plt.savefig('plot_LogP.pdf')

save_3= mannwhitney('LogP')
save_3.to_csv('Mann-Whitney U Test_logP.csv')
save_3

plt.figure(figsize=(10.5, 10.5))

sns.boxplot(x = 'class', y = 'NumHDonors', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=20, fontweight='bold')

plt.savefig('plot_NumHDonors.pdf')

save_3= mannwhitney('NumHDonors')
save_3.to_csv('Mann-Whitney U Test_NumHDoners.csv')
save_3

plt.figure(figsize=(10.5, 10.5))

sns.boxplot(x = 'class', y = 'NumHAcceptors', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=20, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=20, fontweight='bold')

plt.savefig('plot_NumHAcceptors.pdf')

save_4= mannwhitney('NumHAcceptors')
save_4.to_csv('Mann-Whitney U Test_NumHAcceptors.csv')
save_4

! zip -r results.zip . -i *.csv *.pdf

