#Decision trees - titanic

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = sns.load_dataset("titanic")
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df = df.dropna(subset=["age", "sex", "parch", "survived"])

X = df[["age", "parch", "sex", "pclass", "fare"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))

ForestClassifier = RandomForestClassifier(n_estimators=100, max_depth=5)
ForestClassifier.fit(X_test, y_test)
print(ForestClassifier.score(X_test, y_test))
