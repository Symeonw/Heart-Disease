import pandas as pd 
from acquire import acquire_df

def prep_df():
    df = acquire_df()
    df.rename(columns={"trestbps":"blood_pressure", "fbs":"blood_sugar",\
        "thalach": "max_heart_rate"}, inplace=True)
    df.sex = df.sex.astype("category")
    df.cp = df.cp.astype("category")
    df.blood_sugar = df.blood_sugar.astype("category")
    df.restecg = df.restecg.astype("category")
    df.exang = df.exang.astype("category")
    df.slope = df.slope.astype("category")
    df.ca = df.ca.astype("category")
    df.thal = df.thal.astype("category")
    df.target = df.target.astype("category")
    return df

    

def train_test_split(df, train_size = .60):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["target"])
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_size, random_state = 123, stratify=y)
    train = X_train.merge(y_train, left_index = True, right_index=True)
    test = X_test.merge(y_test, left_index=True, right_index=True)
    return train, test

def split_train_and_test(train, test, target):
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]
    return X_train, y_train, X_test, y_test