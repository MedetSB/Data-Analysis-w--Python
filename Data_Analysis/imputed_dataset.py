#Medet Serhat Bingöl
import pandas as pd

imputed_dataset = pd.read_csv('imputed_dataset.csv')
print(imputed_dataset.head())
# Checking for missing values
print(imputed_dataset.isnull().sum())

# Changing type of 'IsVirus' column to numeric
imputed_dataset.iloc[:,-1] = imputed_dataset.iloc[:,-1].astype(int)

# Features and target variability separation
X = imputed_dataset.drop('IsVirus', axis=1)
y = imputed_dataset['IsVirus']

# Data scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Creating Model
model = LogisticRegression(random_state=42)

# Model Training
model.fit(X_train, y_train)

# Prediction on testset
y_pred = model.predict(X_test)

# Performance metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
