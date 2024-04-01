# Loan-Prediction-using-Machine-Learning
Implemented a machine learning-based loan prediction system Random Forest with an accuracy of 81 Percentage  that automates the selection of eligible candidates and Investigated loan data applications to identify patterns and factors that impact loan eligibility, income, and other relevant factors.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Load the dataset
df = pd.read_csv("C:\\Users\\91986\\Desktop\\projects\\loan Eligibility Prediction ,python project for data science\\loan predictive csv\\Loan.csv")

# Check the first few rows of the dataframe
print(df.head())

# Check information about the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Handle missing values
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df.fillna(df.mean(), inplace=True)

# Encode categorical variables
labelencoder = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    df[col] = labelencoder.fit_transform(df[col])

# Split data into features and target variable
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred_rf = rf_clf.predict(X_test)
print("Accuracy of Random Forest Classifier:", metrics.accuracy_score(y_test, y_pred_rf))

# Initialize and train Naive Bayes Classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred_nb = nb_clf.predict(X_test)
print("Accuracy of Naive Bayes Classifier:", metrics.accuracy_score(y_test, y_pred_nb))

# Initialize and train Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred_dt = dt_clf.predict(X_test)
print("Accuracy of Decision Tree Classifier:", metrics.accuracy_score(y_test, y_pred_dt))

# Initialize and train K-Nearest Neighbors Classifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred_knn = knn_clf.predict(X_test)
print("Accuracy of K-Nearest Neighbors Classifier:", metrics.accuracy_score(y_test, y_pred_knn))
