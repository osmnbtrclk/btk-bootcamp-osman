import numpy as np
import pandas as pd 
import pickle
from sklearn.metrics import classification_report
test = pd.read_csv("data/test.csv")
sample_df=test
with open("lbgm.pkl", 'rb') as f:
    model = pickle.load(f)

sample_df = sample_df.loc[:, ~sample_df.columns.isin(['EmployeeCount', 'EmployeeNumber','StandardHours',"Over18"])]

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
scoring = {'accuracy': make_scorer(accuracy_score),
           'prec': 'precision'}


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

change_martial = np.unique(sample_df["Attrition"])
martial_unique ={change_martial[i]: i for i in range(len(change_martial))}
sample_df["Attrition"]=sample_df["Attrition"].map(martial_unique)

change_martial = np.unique(sample_df["Gender"])
martial_unique ={change_martial[i]: i for i in range(len(change_martial))}
sample_df["Gender"]=sample_df["Gender"].map(martial_unique)

change_martial = np.unique(sample_df["JobRole"])
martial_unique ={change_martial[i]: i for i in range(len(change_martial))}
sample_df["JobRole"]=sample_df["JobRole"].map(martial_unique)

change_martial = np.unique(sample_df["OverTime"])
martial_unique ={change_martial[i]: i for i in range(len(change_martial))}
sample_df["OverTime"]=sample_df["OverTime"].map(martial_unique)



sa=model.predict(sample_df)    

accuracy_score(sample_df["Attrition"], sa)
print(classification_report(sample_df["Attrition"], sa))
