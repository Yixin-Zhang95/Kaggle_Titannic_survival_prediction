import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.set_option('display.max_rows', 100, 'display.max_columns', 100)


def missing(df):
    # no missing in "Pclass", "Parch" and "SibSp"
    # fill missing in "Age" by finding median age of similar rows according to Pclass, Parch and SibSp
    age_nan_list = list(df[df['Age'].isnull()].index)
    # print(age_nan_list)
    age_med = df['Age'].median()
    pd.options.mode.chained_assignment = None
    for i in age_nan_list:
        age_pred = df[(df['Pclass'] == df.iloc[i]['Pclass']) & (df['Parch'] == df.iloc[i]['Parch'])
                      & (df['SibSp'] == df.iloc[i]['SibSp'])]['Age'].median()
        # print(age_pred)
        if not np.isnan(age_pred):
            df.loc[i, 'Age'] = age_pred
        else:
            df.loc[i, 'Age'] = age_med

    # fill missing in "Cabin" and "Embarked"
    df['Cabin'] = df['Cabin'].fillna('X')
    df['Embarked'] = df['Embarked'].fillna('S')
    return df


def Pclass(df):
    dic = {1: 3, 2: 2, 3: 1}
    df['Pclass'] = df['Pclass'].map(dic)
    return df


def family(df):
    # add new feature of "Family" based on family size
    df['Family'] = df['Parch'] + df['SibSp'] + 1

    # add "Family" related categorical features
    df['Single'] = df['Family'].map(lambda x: 1 if x == 1 else 0)
    df['Small'] = df['Family'].map(lambda x: 1 if x == 2 else 0)
    df['Medium'] = df['Family'].map(lambda x: 1 if 3 <= x <= 4 else 0)
    df['Large'] = df['Family'].map(lambda x: 1 if x >= 5 else 0)
    return df


def age(df):
    # add "Stage" categorical feature which is related to "Age"
    def stage(x):
        if x <= 4:
            return 'toddler'
        elif 5 <= x <= 12:
            return 'child'
        elif 13 <= x <= 20:
            return 'teen'
        elif 21 <= x <= 40:
            return 'adult'
        elif 41 <= x <= 60:
            return 's_adult'
        else:
            return 'senior'

    df['Stage'] = df['Age'].apply(stage)
    return df


def title(df):
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Title'] = df['Title'].map(Title_Dictionary)
    # drop "Name"
    df = df.drop(['Name'], axis=1)
    return df


def ticket(df):
    def ticket_convert(x):
        if x[0].isdigit():
            return 'X'
        return x.replace(".", "").replace("/", "").strip().split(' ')[0]

    df['Ticket'] = df['Ticket'].apply(ticket_convert)
    # df = df.drop(['Ticket'], axis=1)
    return df


def cabin(df):
    # Extract first character and missing has been replaced with "X"
    df['Cabin'] = df['Cabin'].apply(lambda x: x.split(' ')[0][0])
    return df


def onehot_encode(df):
    obj_cols = []
    for column in df.columns:
        if df[column].dtype == "object":
            obj_cols.append(column)
    print('obj_cols', obj_cols)

    drop_list = []
    for column in obj_cols:
        enc = OneHotEncoder()
        ohe = enc.fit_transform(df[[column]]).toarray()
        cols = [column + "_" + str(enc.categories_[0][i]) for i in range(len(enc.categories_[0]))]
        df_ohe = pd.DataFrame(ohe, columns=cols)
        drop_list.append(column)
        df = pd.concat([df, df_ohe], axis=1)
    df = df.drop(drop_list, axis=1)
    # print('shape after encoding', df.shape)
    return df


def min_max_scaler(df):
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    df = scaler.fit_transform(df)
    # print('scaler', df.shape)
    return df

def standardscaler(df):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df


def process(df):
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Fare'] = np.log(1 + df['Fare'])

    df = missing(df)
    df = Pclass(df)
    df = family(df)
    df = age(df)
    df = title(df)
    df = ticket(df)
    df = cabin(df)
    df = onehot_encode(df)
    # print(df.isnull().sum())
    # print(df.max())

    # df = min_max_scaler(df)
    return df


def family_survival(df):
    df['Last_Name'] = df['Name'].apply(lambda x: str.split(x, ",")[0])
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    DEFAULT_SURVIVAL_VALUE = 0.5
    df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
    df['Family_dead'] = DEFAULT_SURVIVAL_VALUE

    for grp, grp_df in df[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
        if len(grp_df) != 1:
            # A Family group is found.
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0
                if smax == 0:
                    df.loc[df['PassengerId'] == passID, 'Family_dead'] = 1

    for _, grp_df in df.groupby('Ticket'):
        if len(grp_df) != 1:
            for ind, row in grp_df.iterrows():
                if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    passID = row['PassengerId']
                    if smax == 1.0:
                        df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1
                    elif smin == 0.0:
                        df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0
                    if smax == 0:
                        df.loc[df['PassengerId'] == passID, 'Family_dead'] = 1
    return df