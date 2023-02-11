"""Descriptor_Dataset_Preparation.ipynb
Inspired from the following source
https://github.com/dataprofessor/code/blob/master/python/CDD_ML_Part_3_Acetylcholinesterase_Descriptor_Dataset_Preparation.ipynb
Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZBp5i0DCzY7Ts7ETpy07bdR2Bn_HjPJA
"""

! unzip padel.zip

import pandas as pd

df3 = pd.read_csv('GSK3b_chembl2850_04_bioactivity_data_3class_pIC50.csv')

df3

selection = ['canonical_smiles','molecule_chembl_id']
df3_selection = df3[selection]
df3_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)

! cat molecule.smi | head -5

! cat molecule.smi | wc -l

! cat padel.sh

! bash padel.sh

! ls -l

df3_X = pd.read_csv('descriptors_output.csv')

df3_X

df3_X = df3_X.drop(columns=['Name'])
df3_X

df3_Y = df3['pIC50']
df3_Y

dataset3 = pd.concat([df3_X,df3_Y], axis=1)
dataset3

dataset3.to_csv('GSK3b_chembl2850_06_bioactivity_data_3class_pIC50_pubchem_fp.csv', index=False)