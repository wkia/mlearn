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

# 1. Загрузите выборку Boston с помощью функции
# sklearn.datasets.load_boston(). Результатом вызова данной функции
# является объект, у которого признаки записаны в поле data, а целевой
# вектор — в поле target.
inputData = load_boston()
#print inputData
print(inputData.data.shape)

X = inputData.data
#print(X)
print(len(X))
y = inputData.target
#print(y)
print(len(y))

# 2. Приведите признаки в выборке к одному масштабу при помощи функции
# sklearn.preprocessing.scale.
X = scale(X)

# 3. Переберите разные варианты параметра метрики p по сетке от 1 до 10
# с таким шагом, чтобы всего было протестировано 200 вариантов
# (используйте функцию numpy.linspace). Используйте KNeighborsRegressor
# с n_neighbors=5 и weights='distance' — данный параметр добавляет в
# алгоритм веса, зависящие от расстояния до ближайших соседей. В
# качестве метрики качества используйте среднеквадратичную ошибку
# (параметр scoring='mean_squared_error' у cross_val_score; при
# использовании библиотеки scikit-learn версии 0.18.1 и выше необходимо
# указывать scoring='neg_mean_squared_error'). Качество оценивайте, как
# и в предыдущем задании, с помощью кросс-валидации по 5 блокам с
# random_state = 42, не забудьте включить перемешивание выборки
# (shuffle=True).

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = np.array([])
steps = np.linspace(1, 10, 200)
for p in steps:
	kregr = KNeighborsRegressor(p=p, n_neighbors=5, weights='distance', metric='minkowski')
	score = cross_val_score(kregr, cv=kf, X=X, y=y, scoring='neg_mean_squared_error')
	scores = np.append(scores, np.max(score)) # or np.min(score), but not np.mean(score)!!!
	
# 4. Определите, при каком p качество на кросс-валидации оказалось
# оптимальным. Обратите внимание, что cross_val_score возвращает массив
# показателей качества по блокам; необходимо максимизировать среднее
# этих показателей. Это значение параметра и будет ответом на задачу.
print scores
idx = np.argmax(scores)
print "[%d] = %.2f (p=%.2f)" % (idx, scores[idx], steps[idx])

# ===============================
# buildding the answer to submit
answer = "%.1f" % steps[idx]
print(answer)

f = open('answer2-2.txt', 'w')
f.write(answer)
f.close()
