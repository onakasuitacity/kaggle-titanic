import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

def load_data():
    df_train = pd.read_csv("../input/train.csv")
    df_test = pd.read_csv("../input/test.csv")
    df = pd.concat([df_train, df_test]).reset_index(drop=True)
    
    df['Fare'].fillna(df.query('Pclass==3 & Embarked=="S"')['Fare'].median(), inplace=True)
    
    titles = [
        "Mr.", "Miss.", "Mrs.", "Master.", "Dr.", "Rev.", "Col.", "Ms.", 
        "Mlle.", "Mme.", "Capt.", "Countess.", "Major.", "Jonkheer.", "Don.", 
        "Dona.", "Sir.", "Lady."
    ]
    df["Title"] = None
    for title in titles:
        df.loc[df["Name"].str.contains(title, regex=False), "Title"] = title
    df.loc[(df["Title"] != "Master.") & (df["Sex"] == "male"), "Title"] = "Mr."
    df.loc[(df["Title"] != "Mrs.") & (df["Sex"] == "female"), "Title"] = "Miss."
    
    df["Surname_Fare"] = df["Name"].map(lambda name: name.split(',')[0].strip()) + df["Fare"].astype(str)
    df["Count_Surname_Fare"] = df["Surname_Fare"].map(df["Surname_Fare"].value_counts())

    for title in df["Title"].unique():
        df.loc[df["Title"] == title, "Age"].fillna(df.loc[df["Title"] == title, "Age"].mean(), inplace=True)
    
    # ordinal encoding
    oe_columns = [
        "Sex",
        "Title",
        "Pclass",
        "Embarked"
    ]
    oe = ce.OrdinalEncoder(cols=oe_columns, handle_unknown='value')
    df = oe.fit_transform(df)

    
    # extract
    df.drop([
        "Name",
        "Ticket",
        "Cabin"
    ], axis=1, inplace=True)

    df_train, df_test = df[:len(df_train)], df[len(df_train):]
    
    # target encoding
    loo = ce.TargetEncoder(cols="Surname_Fare", handle_unknown="value")
    df_train = loo.fit_transform(df_train, df_train["Survived"])
    df_test = loo.transform(df_test)


if __name__ == "__main__":
    load_data()