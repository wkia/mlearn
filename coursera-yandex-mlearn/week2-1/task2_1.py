# coding=utf-8
import numpy as np
import pandas
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score

## 1. Загрузите выборку Wine по адресу
## https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
## (файл также приложен к этому заданию)
inputData = pandas.read_csv('wine.data', sep=',', header=None)
#print(inputData)

## 2. Извлеките из данных признаки и классы. Класс записан в первом
## столбце (три варианта), признаки — в столбцах со второго по
## последний. Более подробно о сути признаков можно прочитать по адресу
## https://archive.ics.uci.edu/ml/datasets/Wine (см. также файл
## wine.names, приложенный к заданию)
X = inputData.iloc[:, 1:]
#print(X)
print(len(X))
y = inputData.iloc[:, 0]
#print(y)
print(len(y))

## 3. Оценку качества необходимо провести методом кросс-валидации по 5
## блокам (5-fold). Создайте генератор разбиений, который перемешивает
## выборку перед формированием блоков (shuffle=True). Для
## воспроизводимости результата, создавайте генератор KFold с
## фиксированным параметром random_state=42. В качестве меры качества
## используйте долю верных ответов (accuracy).
kf = KFold(n_splits=5, shuffle=True, random_state=42)

##for train_index, test_index in kf.split(X):
##	print("\nTRAIN(",len(train_index),"):", train_index, "\nTEST(",len(test_index),"):", test_index)
##	#X_train, X_test = X[train_index], X[test_index]
##	#y_train, y_test = y[train_index], y[test_index]

## 4. Найдите точность классификации на кросс-валидации для метода k
## ближайших соседей (sklearn.neighbors.KNeighborsClassifier), при k от
## 1 до 50. При каком k получилось оптимальное качество? Чему оно равно
## (число в интервале от 0 до 1)? Данные результаты и будут ответами на
## вопросы 1 и 2.
scores1 = np.array([])
for i in range(1,51):
	neigh = KNeighborsClassifier(n_neighbors=i)
	score = cross_val_score(neigh, cv=kf, X=X, y=y, scoring='accuracy')
	scores1 = np.append(scores1, np.mean(score))

print scores1
idx1 = np.argmax(scores1)
print "[%d] = %.2f" % (idx1, scores1[idx1])

## 5. Произведите масштабирование признаков с помощью функции
## sklearn.preprocessing.scale. Снова найдите оптимальное k на
## кросс-валидации.
X = scale(X)

## 6. Какое значение k получилось оптимальным после приведения признаков
## к одному масштабу? Приведите ответы на вопросы 3 и 4. Помогло ли
## масштабирование признаков?
scores2 = np.array([])
for i in range(1,51):
	neigh = KNeighborsClassifier(n_neighbors=i)
	score = cross_val_score(neigh, cv=kf, X=X, y=y, scoring='accuracy')
	scores2 = np.append(scores2, np.mean(score))

print scores2
idx2 = np.argmax(scores2)
print "[%d] = %.2f" % (idx2, scores2[idx2])

# ===============================
# buildding the answer to submit
answer = ["","","",""]
answer[0] = "%d" % (idx1 + 1)
answer[1] = "%.2f" % scores1[idx1]
answer[2] = "%d" % (idx2 + 1)
answer[3] = "%.2f" % scores2[idx2]
print(answer)

for i in range(0, 4):
	filename = 'answer2-1-%d.txt' % i
	f = open(filename, 'w')
	print answer[i]
	f.write(answer[i])
	f.close()
