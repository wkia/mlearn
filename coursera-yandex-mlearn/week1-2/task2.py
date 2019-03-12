import numpy as np
import pandas
#from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

inputData = pandas.read_csv('titanic.csv', index_col='PassengerId')

# Removing unnecessary columns
inputData = inputData[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]

# Dropping rows containing NaN ('Age' field may contain NaN values)
inputData = inputData.dropna()

# Dividing target from data
data = inputData[['Pclass', 'Fare', 'Age', 'Sex']]
target = inputData['Survived']

# Replacing string values with integer (0-female, 1-male)
#data.loc[data['Sex'] == 'male'] = 1
#data.loc[data['Sex'] == 'female'] = 0
data['Sex'] = (data['Sex'] == 'male').astype(int)
#data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

print(data)
#print(target)

# Train classifier
clf = tree.DecisionTreeClassifier(random_state=241)
clf.fit(data, target)

# Get feature importances
imp = clf.feature_importances_
print(imp)

# Building result data frame
#print(data.columns)
#print(np.vstack((data.columns, imp)).T)
result = pandas.DataFrame(data=np.vstack((data.columns, imp)).T, columns=['Name', 'Importance'])
#print(result)

# Finding two main importances
result = result.sort_values(by='Importance', ascending=False)
print(result)
result = result['Name'].tolist()

# Writing the answer
answer = "%s %s" % (result[0], result[1])
print(answer)
f = open('answer.txt', 'w')
f.write(answer)
f.close()