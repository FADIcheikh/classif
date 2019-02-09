# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



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
#train
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
#score cross_validation
print logreg.score(X_test,y_test)
#test
y_pred = logreg.predict(X_test)
print y_pred
"""
#Transform target to 0,1
for i in range(0,len(target)):
    if target[i] == 'positif':
        target[i] = 1
    else:
        target[i] = 0


X_train, X_test, y_train, y_test = train_test_split(explicative,target, test_size=0.25, random_state=0)
#Roc curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Faux Positifs ')
plt.ylabel('Vrai Positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()"""