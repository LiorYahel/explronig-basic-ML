#Logistic Regression - titanic

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#For machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = sns.load_dataset("titanic")
df.head()
df.info()
df.describe()

df["sex"] = df["sex"].map({"male": 0, "female": 1})
df = df.dropna(subset=["age", "sex", "parch", "survived"])

X = df[["age", "parch", "sex", "pclass", "fare"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
