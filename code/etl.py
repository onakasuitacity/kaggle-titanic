import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

def load_data():
    df_train = pd.read_csv("../input/train.csv")
    df_test = pd.read_csv("../input/test.csv")
    df = pd.concat([df_train, df_test]).reset_index(drop=True)
    
    # missing value
    df['Fare'].fillna(df.query('Pclass==3 & Embarked=="S"')['Fare'].median(), inplace=True)
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    
    # ordinal encoding
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    
    # one hot encoding
    ohe_columns = [
        "Pclass",
        "Embarked"
    ]
    ohe = ce.OneHotEncoder(cols=ohe_columns, handle_unknown='impute')
    df = ohe.fit_transform(df)
    
    # scaling
    sc_columns = [
        "Age",
        "Fare"
    ]
    sc = StandardScaler()
    df[sc_columns] = sc.fit_transform(df[sc_columns])
    
    # extract
    df.drop([
        "Name",
        "Ticket",
        "Cabin"
    ], axis=1, inplace=True)

    
    df_train, df_test = df[:len(df_train)], df[len(df_train):]
    return df_train, df_test


if __name__ == "main":
    load_data()