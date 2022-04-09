import numpy as np
import pandas as pd 
import pickle


# load onehot encoder
with open("preprocessors/onehot_encoder.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

# load scaler
with open("preprocessors/standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# TODO
COLUMNS_TO_REMOVE = ['EmployeeCount', 'EmployeeNumber','StandardHours',"Over18"

]

# TODO
COLUMNS_TO_ONEHOT_ENCODE = ["BusinessTravel", "Department", "EducationField", "MaritalStatus"

]


def preprocess(sample: dict) -> np.ndarray:
    sample_df = pd.DataFrame(sample, index=[0])
    
    sample_df = sample_df.loc[:, ~sample_df.columns.isin(['EmployeeCount', 'EmployeeNumber','StandardHours',"Over18"])]
    
    
    change_travel = np.unique(sample_df["BusinessTravel"])
    business_unique ={change_travel[i]: i for i in range(len(change_travel))}
    sample_df["BusinessTravel"]=sample_df["BusinessTravel"].map(business_unique)
    
    change_deparment = np.unique(sample_df["Department"])
    department_unique ={change_deparment[i]: i for i in range(len(change_deparment))}
    sample_df["Department"]=sample_df["Department"].map(department_unique)
    
    change_education = np.unique(sample_df["EducationField"])
    education_unique ={change_education[i]: i for i in range(len(change_education))}
    sample_df["EducationField"]=sample_df["EducationField"].map(education_unique)
    
    change_martial = np.unique(sample_df["MaritalStatus"])
    martial_unique ={change_martial[i]: i for i in range(len(change_martial))}
    sample_df["MaritalStatus"]=sample_df["MaritalStatus"].map(martial_unique)
    
    week2change=["Gender","JobRole","OverTime"]
    w2_name=[]
    for i in range(len(week2change)):
        print(np.unique(sample_df[week2change[i]]))
        change_w2 = np.unique(sample_df[week2change[i]])
        w2_unique ={change_w2[i]: i for i in range(len(change_w2))}
        w2_name.append([change_w2,w2_unique])
        sample_df[week2change[i]]=sample_df[week2change[i]].map(w2_unique)
    
    # sample_df = drop_columns(sample_df)
    # sample_df = encode_columns(sample_df)
    sample_df = create_features(sample_df)
    # scaled_sample_values = scale(sample_df.values)
    # scaled_sample_values = scaled_sample_values.reshape(1, -1)
    # return scaled_sample_values
    return sample_df

# def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
#     return train=train.loc[:, ~train.columns.isin(['EmployeeCount', 'EmployeeNumber','StandardHours',"Over18"])]



def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    # create a new dataframe with one-hot encoded columns
    encoded_df = pd.DataFrame(onehot_encoder.transform(df[COLUMNS_TO_ONEHOT_ENCODE]).toarray())
    # set new column names
    column_names = onehot_encoder.get_feature_names(COLUMNS_TO_ONEHOT_ENCODE)
    encoded_df.columns = column_names
    # drop raw columns, and add one-hot encoded columns instead
    df = df.drop(columns=COLUMNS_TO_ONEHOT_ENCODE, axis=1)
    df = df.join(encoded_df)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # create MeanAttritionYear feature
    df["MeanAttritionYear"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)

    # TODO
    # bins. pd.IntervalIndex.from_tuples([(-1,5),(5,10) ,(10,15), (15,100)])
    # cat_YearsAtCompany = pd.cut(df["TearsAtCompany"].to_list(),bins)
    # cat_YearsAtCompany.categories=[0,1,2,3]
    # df["YearsAtCompanyCat"]=cat_YearsAtCompany
    return df


def scale(arr: np.ndarray) -> np.ndarray:
    return scaler.transform(arr)
