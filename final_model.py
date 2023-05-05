import os, types

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from  sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import Perceptron
from yellowbrick.classifier import ConfusionMatrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
df = pd.read_csv("D:/ML-Project/dataset/heart.csv")




x = df.drop(['HeartDisease'], axis = 1)
y = df.loc[:,'HeartDisease'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Full_Piplined
cat_attribs =["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
num_attribs = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
full_pipline = ColumnTransformer([("num", StandardScaler(), num_attribs), ("cat", OrdinalEncoder(), cat_attribs)])
data_prepared = full_pipline.fit_transform(x_train)
x_test_pre= full_pipline.transform(x_test)

some_digir = x_test_pre[0]
print(some_digir)
print (y_test[0])
#Traning Models
svc = SVC()
svc.fit(data_prepared,y_train)
predic = svc.predict(x_test_pre)
accu = accuracy_score(y_test,predic)
print("SVC : ",accu)

svc_model = SVC()
svc_model.get_params()
param_grid_svm={'C': [0.1,0.5,1,2,10],'gamma':['scale','auto'],'kernel': ['linear','rbf','poly','sigmoid']}
svc_grid_model= GridSearchCV(svc_model,param_grid_svm,cv=5)
svc_grid_model.fit(data_prepared,y_train)
print(svc_grid_model.best_params_)
svc_pred= svc_grid_model.predict(x_test_pre)
accr = accuracy_score(y_test,svc_pred)
print("SVC with hyperparameter : ",accr)

lr = LogisticRegression()
lr.fit(data_prepared,y_train)
predi = lr.predict(x_test_pre)
accll = accuracy_score(y_test,predi)
print("accuraccy Logis :", accll)

r_forest = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=5,random_state=33)
r_forest.fit(data_prepared,y_train)
predicted = r_forest.predict(x_test_pre)
acc = accuracy_score(y_test,predicted)
print("accuraccyRA :",acc )

DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=33) #criterion can be entropy
DecisionTreeClassifierModel.fit(data_prepared, y_train)
pppp= DecisionTreeClassifierModel.score(x_test_pre, y_test)
print("Dec_tree",pppp)

#The Confusion Matrix
classes = ['Normal', 'Heart_Disease']
r_forest_cm = ConfusionMatrix(r_forest, classes=classes, cmap='GnBu')

r_forest_cm.fit(data_prepared, y_train)
r_forest_cm.score(data_prepared, y_train)
r_forest_cm.show()

# Votting Classifier
voting = VotingClassifier(estimators=[('lr',DecisionTreeClassifierModel), ('rf', r_forest),('svs', svc_model)],voting='hard')
voting.fit(data_prepared,y_train)
vott = cross_val_score(estimator=voting, X=x_test_pre, y=y_test, cv =10)
print(vott)
poiuh = voting.predict(data_prepared)
acc3 = accuracy_score(y_train,poiuh)
print("trgfy  ",acc3)
vott_mean= vott.mean()
print(f"Accuracy of Voting Classifier : {vott.mean()}\n")
print(voting.predict([some_digir]))
#The Final Result
x_Feature=["SVC", "LogisticRegression", "RandomForest", "DecisionTree", "Voting"]
y_Feature=[accr*100,accll*100,acc*100,pppp*100,vott_mean*100]
labels=[accr,accll,acc,pppp,vott_mean]
c=["lightcoral","rosybrown","blanchedalmond","wheat","gray"]
ax=plt.bar(x_Feature,y_Feature,width=0.6,color=c,edgecolor="blue")
plt.xlabel("The Models")
plt.ylabel("Accuracy")

for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.annotate(f'{height/100:%}', (x + width/2, y + height*1.02), ha='center')

plt.show()





