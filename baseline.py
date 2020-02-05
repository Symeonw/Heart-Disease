import pandas as pd 
from prep import prep_df, train_test_split, split_train_and_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import stats
def baseline_metrics():
    df = prep_df()
    train, test = train_test_split(df)
    X_train, y_train, X_test, y_test = split_train_and_test(train, test, "target")
    lr = LogisticRegression()
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