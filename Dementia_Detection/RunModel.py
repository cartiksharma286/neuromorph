import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df_dementia=pd.read_csv('oasis_longitudinal.csv')
df_dementia.head()
# dropping irrelevant columns
df_dementia=df_dementia.drop(['Subject ID','MRI ID','Hand'],axis=1)
df_dementia.head()
df_dementia1=df_dementia
df_dementia.shape
df_dementia.dtypes
df_dementia.describe()
df_dementia.describe(include='object')
df_dementia.isnull().sum()
df_dementia.SES.describe()
df_dementia.SES.mode()
df_dementia.SES.unique()
df_dementia.MMSE.describe()
df_dementia["SES"].fillna(df_dementia["SES"].median(), inplace=True)
df_dementia["MMSE"].fillna(df_dementia["MMSE"].median(), inplace=True)
df_dementia.plot(kind='box',figsize=(12,10),subplots=True,layout=(4,3))
plt.show()
df_dementia.drop(["Visit","MR Delay"],axis=1,inplace=True)
df_dementia["Group"].replace({"Nondemented":0,"Demented":1},inplace=True)
df_dementia["M/F"].replace({"M":0,"F":1},inplace=True)
df_dementia.head(5)
df_dementia["Group"].replace({"Converted":1},inplace=True)
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import update_score_card
df = df_dementia
df["Group"].value_counts()
X=df.drop(["Group"],axis=1)
y=df["Group"]
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
logreg=sm.Logit(y_train,X_train).fit()
logreg.summary()
y_pred=logreg.predict(X_test)
y_pred=[0 if i<0.5 else 1 for i in y_pred]
cm = confusion_matrix(y_pred,y_test)
TP=cm[0,0]
TN=cm[1,1]
FP=cm[1,0]
FN=cm[0,1]

acc = (TN+TP)/(TN+FP+TP+FN)
precision = TP / (TP+FP)
recall = TP / (TP+FN)
specificity = TN / (TN+FP)
f1_score = 2*((precision*recall)/(precision+recall))
target_names = ["1","2"]
cm=classification_report(y_test,y_pred,target_names=target_names)
print(cm)
#update_score_card(algorithm_name = 'Logistic Regression', model = logreg)
#print(score_card)
from sklearn.feature_selection import RFE
X=df.drop(["Group"],axis=1)
y=df["Group"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 10)

linreg_rfe = LogisticRegression()

rfe_model = RFE(estimator=linreg_rfe, n_features_to_select = 6)

rfe_model = rfe_model.fit(X_train, y_train)

feat_index = pd.Series(data = rfe_model.ranking_, index = X_train.columns)

signi_feat_rfe = feat_index[feat_index==1].index

print(signi_feat_rfe)

X2=df[['EDUC', 'SES', 'nWBV', 'ASF', 'MMSE', 'CDR']]
y2=df["Group"]
X2=sm.add_constant(X2)
X_train,X_test,y_train,y_test=train_test_split(X2,y2,test_size=0.3)

logreg=sm.Logit(y_train,X_train).fit()
logreg.summary()
y_pred=logreg.predict(X_test)
correct=(TN+TP)/(TN+TP+FP+FN)
print("Correctly classified :",correct*100)
in_correct=(FN+FP)/(TN+TP+FP+FN)
print("In_Correctly classified :",in_correct*100)
#update_score_card(algorithm_name = 'Logistic Regression -RFE', model = logreg)
from sklearn.metrics import plot_confusion_matrix

#score_card
#cm=classification_report(y_test,y_pred)
#print(cm)
#plot_confusion_matrix(gnb_model)
#cm = confusion_matrix(y_pred,y_test)
#TP=cm[0,0]
#TN=cm[1,1]
#FP=cm[1,0]
#FN=cm[0,1]

acc = (TN+TP)/(TN+FP+TP+FN)
precision = TP / (TP+FP)
recall = TP / (TP+FN)
specificity = TN / (TN+FP)
f1_score = 2*((precision*recall)/(precision+recall))
#update_score_card(algorithm_name = 'Naive Bayes', model = gnb_model)
#score_card
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
knn_classification = KNeighborsClassifier(n_neighbors = 3)
knn_model = knn_classification.fit(X_train, y_train)
y_pred=knn_model.predict(X_test)
#plot_confusion_matrix(knn_model)
cm = confusion_matrix(y_pred,y_test)
TP=cm[0,0]
TN=cm[1,1]
FP=cm[1,0]
FN=cm[0,1]

acc = (TN+TP)/(TN+FP+TP+FN)
precision = TP / (TP+FP)
recall = TP / (TP+FN)
specificity = TN / (TN+FP)
f1_score = 2*((precision*recall)/(precision+recall))
#update_score_card(algorithm_name = 'KNN ', model = knn_model)
from sklearn.ensemble import GradientBoostingClassifier
#score_card
gboost_model = GradientBoostingClassifier(n_estimators = 150, max_depth = 10, random_state = 10)

gboost_model.fit(X_train, y_train)
y_pred=gboost_model.predict(X_test)
#plot_confusion_matrix(gboost_model)
#test_report = get_test_report(gboost_model)
from xgboost import XGBClassifier
#print(test_report)
xgb_model = XGBClassifier(max_depth = 10, gamma = 1)

xgb_model.fit(X_train, y_train)
y_pred=xgb_model.predict(X_test)
#plot_confusion_matrix(xgb_model)
