# coding=utf-8
import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import roc_auc_score

# 1. Загрузите данные из файла data-logistic.csv. Это двумерная выборка,
# целевая переменная на которой принимает значения -1 или 1.  

data = pd.read_csv('data-logistic.csv', header=None)
x = data.iloc[:,1:] # столбцы со второго до конца (т.е. 2й и 3й)
y = data.iloc[:,0] # первый столбец

# 2. Убедитесь, что выше выписаны правильные формулы для градиентного спуска.
# Обратите внимание, что мы используем полноценный градиентный спуск, а не его
# стохастический вариант!

def gradStep(x, y, w, C):
	k = 0.1 # шаг логистической регрессии
	N = len(y)
	A = -y * np.dot(x, w)
	w_res = np.array(w).copy()
	for j in range(len(w)):
		S = 0
		for i in range(N):
			S += y[i] * x.iloc[i,j] * (1 - 1 / (1 + math.exp(A[i])))
		w_res[j] += k / N * S - k * C * w_res[j]
	return w_res

# 3. Реализуйте градиентный спуск для обычной и L2-регуляризованной (с
# коэффициентом регуляризации 10) логистической регрессии. Используйте длину
# шага k=0.1. В качестве начального приближения используйте вектор (0, 0).

def gradImpl(x, y, C=0.):
	N = 10000 # Max iterations
	E = 1e-5 # Error threshold

	w = np.zeros(x.shape[1]) # начальное приближение
	i = 0
	while i < N:
		w_new = gradStep(x, y, w, C)
		e = math.sqrt(((w_new - w) ** 2).sum())
		if e <= E:
			break
		i += 1
		w = w_new
	return w

# 4. Запустите градиентный спуск и доведите до сходимости (евклидово расстояние
# между векторами весов на соседних итерациях должно быть не больше 1e-5).
# Рекомендуется ограничить сверху число итераций десятью тысячами.

w = gradImpl(x, y)
wr = gradImpl(x, y, C=10.) # с коэффициентом регуляризвции

# 5. Какое значение принимает AUC-ROC на обучении без регуляризации и при ее
# использовании? Эти величины будут ответом на задание. В качестве ответа
# приведите два числа через пробел. Обратите внимание, что на вход функции
# roc_auc_score нужно подавать оценки вероятностей, подсчитанные обученным
# алгоритмом. Для этого воспользуйтесь сигмоидной функцией: 
# a(x) = 1 / (1 + exp(-w1 x1 - w2 x2)). 

def sigmoid(x, w):
	return 1 / (1 + math.exp( -(x * w).sum() ))

y_proba = x.apply(lambda x: sigmoid(x, w), axis=1)
auc = roc_auc_score(y, y_proba)

yr_proba = x.apply(lambda x: sigmoid(x, wr), axis=1)
rauc = roc_auc_score(y, yr_proba)

# ===============================
# buildding the answer to submit
answer = '%.3f %.3f' % (auc, rauc)
print('answer=', answer)
f = open('answer3-3.txt', 'w')
f.write(answer)
f.close()

