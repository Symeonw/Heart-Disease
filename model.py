from prep import prep_df, train_test_split, split_train_and_test
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from baseline import baseline_metrics
import pandas as pd
import numpy as np
baseline_metrics()

df = prep_df()
df.drop(columns=["blood_sugar", "age"], inplace=True) #Dropping due to conclusions reached in exploration.
train, test = train_test_split(df, train_size=.50)
X_train, y_train, X_test, y_test = split_train_and_test(train, test, "target")

lr = LogisticRegression(random_state=123)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
lr.score(X_train, y_train)
y_pred_test = lr.predict(X_test)
acc_train = round(lr.score(X_train, y_train) * 100, 2)
acc_test = round(lr.score(X_test, y_test) * 100, 2)
class_report = classification_report(y_test, y_pred_test)
print(class_report)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_test)
print(f"Accuracy on Train set is: {acc_train}%")
print (f"Accuracy on Test set is: {acc_test}%")
print("Confusion Matrix: False Neg: 20, False Pos: 13.")

#Conclusion:
# Dropping blood_sugar and age contributed to the increase of accuracy by 1.75%, 
#   bringing accuracy on test data (with .5 test-train-split) to 86.18%. 

# Random Forest Test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = df.iloc[:,:-1], df.iloc[:,-1]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
clf = RandomForestClassifier(max_depth=30, random_state=123)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)



#XGBoost Test

import xgboost as xgb
X, y = df.iloc[:,:-1], df.iloc[:,-1]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.5, random_state=123, stratify=y)


xg_cl = xgb.XGBClassifier(objective="binary:logistic", n_estimators=10, seed=123)

xg_cl.fit(X_train, y_train)
xg_cl.score(X_test, y_test)

#Second 

hd_dmatrix = xgb.DMatrix(data=X, label=y)
params={"objective": "binary:logistic", "max_depth":3}
cv = xgb.cv(dtrain=hd_dmatrix, params=params, nfold=4,\
    num_boost_round=10, metrics="auc", as_pandas=True)
print(cv)
print((cv["test-auc-mean"]).iloc[-1])


