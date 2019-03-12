import pandas
import numpy as np
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
 
print(data.axes)
#print(data['Pclass'].value_counts())
 
# answer1
countBySex = data['Sex'].value_counts()
answer1 = "%d %d" % (countBySex['male'], countBySex['female'])
print(answer1)
f = open('answer1.txt', 'w')
f.write(answer1)
f.close()
 
# answer2
survived = data['Survived'].value_counts()
#print(survived)
#print(len(data.index))
answer2 = "%.2f" % (100.0 * survived[1] / data['Survived'].count())
print(answer2)
f = open('answer2.txt', 'w')
f.write(answer2)
f.close()
 
# answer3
classes = data['Pclass'].value_counts()
#print(classes[1])
answer3 = "%.2f" % (100.0 * classes[1] / data['Pclass'].count())
print(answer3)
f = open('answer3.txt', 'w')
f.write(answer3)
f.close()
 
# answer4
#print(data['Age'])
ageFiltered = data['Age'].dropna()
#print(ageFiltered)
ageMean = np.mean(ageFiltered)
ageMedian = np.median(ageFiltered)
answer4 = "%.2f %.2f" % (ageMean, ageMedian)
print(answer4)
f = open('answer4.txt', 'w')
f.write(answer4)
f.close()
 
# answer5
corr = data['SibSp'].corr(data['Parch'])
answer5 = "%.2f" % corr;
print(answer5)
f = open('answer5.txt', 'w')
f.write(answer5)
f.close()
 
# answer6
femaleNames = data[data.Sex == 'female']['Name']
#firstnames: data['firstnames'] = data.composers.str.split('\s+').str[0]
#lastnames: data.composers.str.split('\s+').str[-1]
#all but the lastnames: data.composers.str.split('\s+').str[:-1].apply(lambda parts: " ".join(parts))
#print(femaleNames)
firstnamesMrs = (femaleNames.str.split('\((.*)\)').str[1]).dropna()
firstnamesMiss = (femaleNames.str.split(', Miss.\s+(.*)').str[1]).dropna()
firstnamesMs = (femaleNames.str.split(', Ms.\s+(.*)').str[1]).dropna()
firstnamesMlle = (femaleNames.str.split(', Mlle.\s+(.*)').str[1]).dropna()
#
#    2      Cumings, Mrs. John Bradley (Florence Briggs Th...
#    3      Heikkinen, Miss. Laina
#    370,1,1,"Aubart, Mme. Leontine Pauline",female,24,0,0,PC 17477,69.3,B35,C
#    444,1,2,"Reynaldo, Ms. Encarnacion",female,28,0,0,230434,13,,S
#    642,1,1,"Sagesser, Mlle. Emma",female,24,0,0,PC 17477,69.3,B35,C
#    760,1,1,"Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)",female,33,0,0,110152,86.5,B77,S
#??? 797,1,1,"Leader, Dr. Alice (Farnham)",female,49,0,0,17465,25.9292,D17,S
#
firstnames = pandas.concat([firstnamesMiss, firstnamesMrs, firstnamesMs, firstnamesMlle]).str.split('\s+').str[0]
#print(firstnames)
firstnames = firstnames.value_counts()
#print(firstnames)
answer6 = firstnames.index[0]
print(answer6)
f = open('answer6.txt', 'w')
f.write(answer6)
f.close()
