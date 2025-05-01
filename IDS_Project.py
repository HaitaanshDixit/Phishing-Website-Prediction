import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

file = r'.vscode\Projects\phishing_website_dataset.csv'   # Importing file through relative path
df = pd.read_csv(file)                                    # Converting file dataset into dataframe
print(df.head(), '\n\n')

# Splitting features and target
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]


# Identifying non-numeric columns (for using logisitic regression)
non_numeric_cols = X.select_dtypes(include=['object']).columns
print(f"Non-numeric col: {non_numeric_cols}", '\n\n')
X = X.drop(columns=non_numeric_cols)



#PRE_PROCESSING

print(df.isnull().sum(), '\n\n')   # There is no null value in any column thus we do not need to do imputation using SimpleImputer

# Data Encoding (but from looking at our dataset, there is no need of encoding as non-numeric data is there very little i.e. only for websites names and domain url

# Data Scaling
from sklearn.preprocessing import StandardScaler
# Selecting numerical columns excluding the target column 'label'
num_cols = X.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])


# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



# Visualizing before Classification

# Histograms
df.hist(figsize=(18, 18))
plt.suptitle("Feature Distributions", fontsize=7)
plt.show()

# Scatter Plots for Relationships
#sns.pairplot(df, hue='label')     
#plt.suptitle("Features", fontsize=4)
#plt.show()

# Correlation Matrix
#plt.figure(figsize=(8,8))
#sns.heatmap(new_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", cbar=True, linewidths=0.5)
#plt.title("Correlation Matrix")
#plt.show()

# Counting labels
plt.figure(figsize=(8, 8))
sns.countplot(y='label', data=df, palette="Set1", hue='label', dodge=False)
plt.title("Count of label data")
plt.show()


"""
from sklearn.ensemble import GradientBoostingClassifier
# (We will use GBC because it may be slow, but it can easily handle large and complex data)

model = GradientBoostingClassifier(random_state=42)

# HyperParameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.05],
    'max_depth': [3, 4, 5, 6, 7]
             }
# Perform GridSearch
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# cv is cross validation, in param grid, 5 values each are given so cv is 5 and njobs is for cpu usage, 1 uses 1 cpu, -1 uses all cpu, 2 and so on use that many cpu cores.

# Perform the grid search
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}", '\n\n')

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print(y_pred, '\n\n')
"""
# GBC is very slow over this large dataset (on my laptop), so we will try to use Logistic regression now, main requirment for that is dropping non-numeric columns so that it can be used, so I did it above before preprocessing.


# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred, "\n\n")

# Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score = {acc}", '\n\n')



# Visualize After Classification

from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Classification Report
print('\n\n',"Classification Report:\n", classification_report(y_test, y_pred))





