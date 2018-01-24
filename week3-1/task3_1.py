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
from sklearn.svm import SVC

## 1. Загрузите выборку из файла svm-data.csv. В нем записана двумерная 
## выборка (целевая переменная указана в первом столбце, признаки — 
## во втором и третьем).

inputData = pandas.read_csv('svm-data.csv', header=None)
X = inputData.iloc[:, 1:]
print(np.shape(X))
y = inputData.iloc[:, 0]
print(np.shape(y))

#inputData = pandas.read_csv('perceptron-test.csv', header=None)
#X_test = inputData.iloc[:, 1:]
#print(np.shape(X_test))
#y_test = inputData.iloc[:, 0]
#print(np.shape(y_test))

## 2. Обучите классификатор с линейным ядром, параметром C = 100000 и 
## random_state=241. Такое значение параметра нужно использовать, чтобы 
## убедиться, что SVM работает с выборкой как с линейно разделимой. При более 
## низких значениях параметра алгоритм будет настраиваться с учетом слагаемого
## в функционале, штрафующего за маленькие отступы, из-за чего результат может
## не совпасть с решением классической задачи SVM для линейно разделимой 
## выборки.

clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(X, y)

## 3. Найдите номера объектов, которые являются опорными (нумерация с единицы).
## Они будут являться ответом на задание. Обратите внимание, что в качестве 
## ответа нужно привести номера объектов в возрастающем порядке через запятую 
## или пробел. Нумерация начинается с 1.
## (Индексы опорных объектов обученного классификатора хранятся в поле support_)

f = ' '.join(['%d'] * len(clf.support_))
answer = f % tuple(clf.support_ + 1)
print(answer)

# ===============================
# buildding the answer to submit
f = open('answer3-1.txt', 'w')
f.write(answer)
f.close()

