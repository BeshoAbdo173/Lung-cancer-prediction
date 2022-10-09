
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('D:\6th Term\Artificial Intellegence\12th Project\Phase 1\archive (2)/lung_cancer.csv')
print('Dataset :',data.shape)
data.info()
data[0:10]

data.Result.value_counts()[0:30].plot(kind='bar')
plt.show()

sns.set_style("whitegrid")
sns.pairplot(data,hue="Result",size=3);
plt.show()

data1 = data.drop(columns=['Name','Surname'],axis=1)
data1 = data1.dropna(how='any')
print(data1.shape)

print(data1.shape)
data1.head()

from sklearn.model_selection import train_test_split
Y = data1['Result']
X = data1.drop(columns=['Result'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=9)

print('X train shape: ', X_train.shape)
print('Y train shape: ', Y_train.shape)
print('X test shape: ', X_test.shape)
print('Y test shape: ', Y_test.shape)

from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# We define the SVM model
svmcla = OneVsRestClassifier(BaggingClassifier(SVC(C=10,kernel='linear',random_state=9, probability=True), n_jobs=-1))

# We train model
svmcla.fit(X_train, Y_train)

# We predict target values
Y_predict2 = svmcla.predict(X_test)

# The confusion matrix
from sklearn.metrics import confusion_matrix
svmcla_cm = confusion_matrix(Y_test, Y_predict2)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(svmcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('SVM Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

# Test score
score_svmcla = svmcla.score(X_test, Y_test)
print(score_svmcla)