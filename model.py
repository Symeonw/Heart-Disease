from prep import prep_df, train_test_split, split_train_and_test
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

import xgboost as xgb
to_int = ["sex", "cp", "restecg", "exang", "slope", "ca", "thal", "target"]
for n in to_int:
    df[n] = df[n].astype(int)
X = df.drop(columns=["target"])
y = df.target
df2 = pd.get_dummies(X)
df2 = df2.join(y)
train, test = train_test_split(df2, train_size=.50)
X_train, y_train, X_test, y_test = split_train_and_test(train, test, "target")

xgb_model = xgb.XGBRFRegressor(random_state=42).fit(X_train, y_train)
xgb_model.score(X_test, y_test)


param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, train, num_round)
# make prediction
preds = bst.predict(test)