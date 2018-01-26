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
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

def doTask():
	## 1. Загрузите объекты из новостного датасета 20 newsgroups, относящиеся  
	## к категориям "космос" и "атеизм". Обратите внимание, что загрузка данных 
	## может занять несколько минут.

	print('Loading data...')
	newsgroups = datasets.fetch_20newsgroups(
						subset='all', 
						categories=['alt.atheism', 'sci.space'])
	print('...done')
	#print(dir(newsgroups))
	#print(newsgroups.filenames)
	X = newsgroups.data
	y = newsgroups.target
	#print(np.shape(X))
	#print(np.shape(y))

	## 2. Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в 
	## этом задании мы предлагаем вам вычислить TF-IDF по всем данным. При таком 
	## подходе получается, что признаки на обучающем множестве используют 
	## информацию из тестовой выборки — но такая ситуация вполне законна, 
	## поскольку мы не используем значения целевой переменной из теста. На 
	## практике нередко встречаются ситуации, когда признаки объектов тестовой 
	## выборки известны на момент обучения, и поэтому можно ими пользоваться при 
	## обучении алгоритма.				

	vzer = TfidfVectorizer()
	V = vzer.fit_transform(X)
	#print(V)
	#print(np.shape(V))
	#print(np.shape(vzer.get_feature_names()))

	## 3. Подберите минимальный лучший параметр C из множества 
	## [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром (kernel='linear') 
	## при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 
	## и для SVM, и для KFold. В качестве меры качества используйте долю верных 
	## ответов (accuracy).

	grid = {'C': np.power(10.0, np.arange(-5, 6))}
	cv = KFold(n_splits=5, shuffle=True, random_state=241)
	clf = SVC(kernel='linear', random_state=241)
	gs = GridSearchCV(estimator=clf, param_grid=grid, scoring='accuracy', cv=cv, n_jobs=-1)
	v = vzer.transform(X)
	#print(v)
	#print(np.shape(v))
	
	print('GridSearchCV fitting...')
	gs.fit(V, y)
	print('...done')
	#print(gs.cv_results_)
	#print(dir(gs.cv_results_))
	
	print("-----------")
	print(gs.best_estimator_)
	print(gs.best_score_)
	print(gs.best_index_)
	print(gs.best_params_)
	print("-----------")
	
	best_c = gs.best_params_['C']
	
	## 4. Обучите SVM по всей выборке с оптимальным параметром C, найденным на 
	## предыдущем шаге.

	clf = SVC(kernel='linear', C=best_c, random_state=241)
	clf.fit(V, y)
	
	## 5. Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся 
	## в поле coef_ у svm.SVC). Они являются ответом на это задание. Укажите 
	## эти слова через запятую или пробел, в нижнем регистре, в 
	## лексикографическом порядке.
	
	N = 10
	coef = pandas.DataFrame(clf.coef_.data, clf.coef_.indices)
	coef = coef[0].map(lambda weight: abs(weight))
	coef = coef.sort_values(ascending=False).head(N)
	print(coef)
	
	words = vzer.get_feature_names()
	words = coef.index.map(lambda i: words[i])
	
	answer = ' '.join(words.sort_values())
	print(answer)

	# ===============================
	# buildding the answer to submit
	f = open('answer3-2.txt', 'w')
	f.write(answer)
	f.close()

if __name__ == '__main__':
	doTask()
