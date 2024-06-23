draimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv('/content/ASD_data_csv.csv')

# Preprocess the data
X = data.drop('ASD_traits', axis=1)
y = data['ASD_traits']

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)
y_pred = rf_clf.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Visualization
# Histogram of Qchat_10_Score
plt.figure(figsize=(8, 6))
data['Qchat_10_Score'].hist(bins=10)
plt.xlabel('Qchat_10_Score')
plt.ylabel('Count')
plt.title('Histogram of Qchat_10_Score')
plt.show()

# Scatter plot of Social_Responsiveness_Scale vs Age_Years
plt.figure(figsize=(8, 6))
plt.scatter(data['Age_Years'], data['Social_Responsiveness_Scale'])
plt.xlabel('Age_Years')
plt.ylabel('Social_Responsiveness_Scale')
plt.title('Scatter plot of Social_Responsiveness_Scale vs Age_Years')
plt.show()

# Histogram of Age_Years
plt.figure(figsize=(8, 6))
data['Age_Years'].hist(bins=10)
plt.xlabel('Age_Years')
plt.ylabel('Count')
plt.title('Histogram of Age_Years')
plt.show()

# Box plot of Age_Years
plt.figure(figsize=(8, 6))
data.boxplot(column=['Age_Years'])
plt.xlabel('Age_Years')
plt.title('Box plot of Age_Years')
plt.show()
