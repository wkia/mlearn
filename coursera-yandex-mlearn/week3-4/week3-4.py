# coding=utf-8
import math
import numpy as np
import pandas as pd
import sklearn.metrics as sklm

# 1. Загрузите файл classification.csv. 
# В нем записаны истинные классы объектов выборки (колонка true) 
# и ответы некоторого классификатора (колонка pred).
data = pd.read_csv('classification.csv') # with header

# 2. Заполните таблицу ошибок классификации: 	
# Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям. 
# Например, FP — это количество объектов, имеющих класс 0, но отнесенных 
# алгоритмом к классу 1. Ответ в данном вопросе — четыре числа через пробел.
TP = len(data[(data['true'] == 1) & (data['pred'] == 1)])
FP = len(data[(data['true'] == 0) & (data['pred'] == 1)])
TN = len(data[(data['true'] == 0) & (data['pred'] == 0)])
FN = len(data[(data['true'] == 1) & (data['pred'] == 0)])

# buildding the answer to submit
answer = '%d %d %d %d' % (TP, FP, FN, TN)
print('answer=', answer)
f = open('answer3-4-1.txt', 'w')
f.write(answer)
f.close()

# 3. Посчитайте основные метрики качества классификатора:
# Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
# Precision (точность) — sklearn.metrics.precision_score
# Recall (полнота) — sklearn.metrics.recall_score
# F-мера — sklearn.metrics.f1_score
# В качестве ответа укажите эти четыре числа через пробел.
A = sklm.accuracy_score(data['true'], data['pred'])
P = sklm.precision_score(data['true'], data['pred'])
R = sklm.recall_score(data['true'], data['pred'])
F = sklm.f1_score(data['true'], data['pred'])

# buildding the answer to submit
answer = '%.2f %.2f %.2f %.2f' % (A, P, R, F)
print('answer=', answer)
f = open('answer3-4-2.txt', 'w')
f.write(answer)
f.close()

# 4. Имеется четыре обученных классификатора. В файле scores.csv записаны 
# истинные классы и значения степени принадлежности положительному классу 
# для каждого классификатора на некоторой выборке:
# для логистической регрессии — вероятность положительного класса (колонка score_logreg),
# для SVM — отступ от разделяющей поверхности (колонка score_svm),
# для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
# для решающего дерева — доля положительных объектов в листе (колонка score_tree).
# Загрузите этот файл.
data = pd.read_csv('scores.csv') # with header

# 5. Посчитайте площадь под ROC-кривой для каждого классификатора. 
# Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)? 
# Воспользуйтесь функцией sklearn.metrics.roc_auc_score.
scores = {'score_logreg' : 0., 'score_svm' : 0., 'score_knn' : 0., 'score_tree' : 0.}
for name, _ignored in scores.items():
    scores[name] = sklm.roc_auc_score(data['true'], data[name])

res = pd.Series(scores).sort_values()

# buildding the answer to submit
answer = res.tail(1).index[0]
print('answer=', answer)
f = open('answer3-4-3.txt', 'w')
f.write(answer)
f.close()

# 6. Какой классификатор достигает наибольшей точности (Precision) 
# при полноте (Recall) не менее 70% ? 
# Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой 
# с помощью функции sklearn.metrics.precision_recall_curve. Она возвращает три 
# массива: precision, recall, thresholds. В них записаны точность и полнота при 
# определенных порогах, указанных в массиве thresholds. Найдите максимальной 
# значение точности среди тех записей, для которых полнота не меньше, чем 0.7.

precisions = {'score_logreg' : 0., 'score_svm' : 0., 'score_knn' : 0., 'score_tree' : 0.}
for name, _ignored in precisions.items():
    pr = sklm.precision_recall_curve(data['true'], data[name])
    pr = pd.DataFrame({'precision': pr[0], 'recall': pr[1]})
    pr = pr[pr['recall'] >= 0.7]['precision']
    precisions[name] = pr.max()

res = pd.Series(precisions).sort_values()

# buildding the answer to submit
answer = res.tail(1).index[0]
print('answer=', answer)
f = open('answer3-4-4.txt', 'w')
f.write(answer)
f.close()

