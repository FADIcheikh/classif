# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



df = pd.read_csv("D:\Data_Minig\seance8_classif_KNN_SVM_regLog\\accouchement_premature_donnees.csv",sep =';',header = 0)
#split dataset into explicatives vars and target
explicative =df.drop(['PREMATURE'],axis=1)
names = explicative.columns
target =df['PREMATURE']
#set 25% for test
X_train, X_test, y_train, y_test = train_test_split(explicative,target, test_size=0.25, random_state=0)
#Centrage et reduction
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#check for NA :there is no NA
for index, row in explicative.iterrows():
    for j in  explicative.columns:
      if(np.isnan(row[j])):
        print index
        print j+ " "+str(row[j])
#train k choosen randomly at 1st
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)
#score cross_validation
print classifier.score(X_test,y_test)
#test
y_pred = classifier.predict(X_test)
print y_pred
#calculating error to get the best K
error = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
#displaying mean error for k in 1..20
plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error / K Value')
plt.xlabel('K ')
plt.ylabel(' Erreur Moyenne')
#Best k =7
plt.show()
