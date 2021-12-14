import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# import dataset
dataset = pd.read_csv("data.csv")

X = dataset.iloc[:-1, 2:-1].values
y = dataset.iloc[:-1, 1].values
z = dataset.iloc[-1, 2:-1].values
Z_res = dataset.iloc[-1, 1]
# Le encoding
le = LabelEncoder()
y = le.fit_transform(y)
# z_res = le.fit_transform(Z_res)
# data test set split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standarization

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Random Forest
classifier = RandomForestClassifier(n_estimators=500)
classifier.fit(X_train, y_train)

results = classifier.predict(X_test)

counter = 0
for i in range(len(y_test)):
    if y_test[i] - results[i] > 0:
        print(y_test[i], results[i])
        counter += 1

size = X_train.shape
print(f"FAILS: {counter/size[0]*100}%")
