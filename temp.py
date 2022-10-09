

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv('lung_cancer.csv')
print('Dataset :',data.shape)
data.info()


data.Result.value_counts()[0:30].plot(kind='bar')
plt.show()

sns.set_style("whitegrid")
sns.pairplot(data,hue="Result",height=3);
plt.show()

data1 = data.drop(columns=['Name','Surname'],axis=1)
data1 = data1.dropna(how='any')




from sklearn.model_selection import KFold
Y = data1['Result']
X = data1.drop(columns=['Result'])
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

kfold = KFold(n_splits = 10, shuffle = True, random_state = (1))


from numpy import mean
from numpy import std
from sklearn.svm import SVC

# We define the SVM model
svmcla = SVC(kernel='linear')

# We train model
from sklearn.model_selection import cross_val_score
score = cross_val_score(svmcla, X, Y, scoring='accuracy', cv=kfold, n_jobs=-1)
scores2 = cross_val_score(svmcla, X, Y, scoring='precision', cv=kfold, n_jobs=-1)
scores3 = cross_val_score(svmcla, X, Y, scoring='recall', cv=kfold, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(score), std(score)))
print('Precision: %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall: %.3f (%.3f)' % (mean(scores3), std(scores3)))

# We define the SVM model



# We train model

#svmcla.fit(X_train, Y_train)

# We predict target values

#Y_predict2 = svmcla.predict(X_test)

# The confusion matrix
'''
svmcla_cm = confusion_matrix(Y_test, Y_predict2)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(svmcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('SVM Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()
# Test score

score_svmcla_1 = accuracy_score(Y_test, Y_predict2)
score_svmcla_2 = precision_score(Y_test, Y_predict2)
score_svmcla_3 = recall_score(Y_test, Y_predict2)
score_svmcla_4 = svmcla_cm[1,1]/((svmcla_cm[1,0]+svmcla_cm[1,1]))

print('Accuracy : ',score_svmcla_1)
print('Precision : ',score_svmcla_2)
print('Sensitivity : ',score_svmcla_3)
print('Specificity : ',score_svmcla_4)
'''

























































