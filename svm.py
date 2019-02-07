# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt



df = pd.read_csv("D:\Data_Minig\seance8_classif_KNN_SVM_regLog\\accouchement_premature_donnees.csv",sep =';',header = 0)
#split dataset into explicatives vars and target
explicative =df.drop(['PREMATURE'],axis=1)
names = explicative.columns
target =df['PREMATURE']
#set 25% for test
X_train, X_test, y_train, y_test = train_test_split(explicative,target, test_size=0.25, random_state=0)
#check for NA :there is no NA
for index, row in explicative.iterrows():
    for j in  explicative.columns:
      if(np.isnan(row[j])):
        print index
        print j+ " "+str(row[j])
#train
clf = svm.SVC(gamma='scale', kernel ='linear',decision_function_shape='ovo')
clf.fit(X_train, y_train)
#score cross_validation
print clf.score(X_test,y_test)
#test
print clf.predict([[31,3,100,3,2,2,28,3,2,0,2,1,1],[31,3,58,3,2,1,38,3,2,0,2,1,1],[30,0,25,3,1,2,32,3,3,1,2,2,1]])
#plot nuage de points
"""
plt.scatter(df['DIAB'],target,s=100)

plt.title('Nuage de points')
plt.xlabel('Diabete')
plt.ylabel('Permature')
plt.show()
"""