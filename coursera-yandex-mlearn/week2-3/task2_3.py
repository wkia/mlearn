# coding=utf-8
import numpy as np
import pandas
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

## 1. Загрузите обучающую и тестовую выборки из файлов
## perceptron-train.csv и perceptron-test.csv. Целевая переменная
## записана в первом столбце, признаки — во втором и третьем.

inputData = pandas.read_csv('perceptron-train.csv', header=None)
X = inputData.iloc[:, 1:]
print(np.shape(X))
y = inputData.iloc[:, 0]
print(np.shape(y))

inputData = pandas.read_csv('perceptron-test.csv', header=None)
X_test = inputData.iloc[:, 1:]
print(np.shape(X_test))
y_test = inputData.iloc[:, 0]
print(np.shape(y_test))

## 2. Обучите персептрон со стандартными параметрами и random_state=241.

clf = Perceptron(random_state=241)
clf.fit(X, y)

## 3. Подсчитайте качество (долю правильно классифицированных объектов,
## accuracy) полученного классификатора на тестовой выборке.

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

## 4. Нормализуйте обучающую и тестовую выборку с помощью класса
## StandardScaler.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

## 5. Обучите персептрон на новой выборке. Найдите долю правильных
## ответов на тестовой выборке.

clf = Perceptron(random_state=241)
clf.fit(X_scaled, y)
y_pred = clf.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred)
print(accuracy_scaled)

## 6. Найдите разность между качеством на тестовой выборке после
## нормализации и качеством до нее. Это число и будет ответом на
## задание.

answer = "%.3f" % (accuracy_scaled - accuracy)
print(answer)

# ===============================
# buildding the answer to submit
f = open('answer2-3.txt', 'w')
f.write(answer)
f.close()

