#Medet Serhat Bingöl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
# Reading the CSV file(dataset)
dataset = pd.read_csv('dataset.csv')
# Checking for missing values
print(dataset.isnull().sum())
# Histogram plotting
sns.histplot(dataset,kde=True)
plt.show()
# Imputation
imputed_dataset = dataset.iloc[:,0:-1]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed_dataset = imputer.fit_transform(imputed_dataset)
# Previous command line turns a numpy array. So we must change it as a pandas dataframe
imputed_dataset = pd.DataFrame(imputed_dataset)
imputed_dataset = imputed_dataset.assign(IsVirus=dataset.iloc[:,-1])
sns.histplot(imputed_dataset,kde=True)
plt.show()
print(imputed_dataset.isnull().sum())

#Updating to a new CSV file
dataset.to_csv('imputed_dataset.csv' , index=True)