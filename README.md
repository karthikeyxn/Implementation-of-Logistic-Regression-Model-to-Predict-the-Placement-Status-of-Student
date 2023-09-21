# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Find new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: R.Guruprasad
RegisterNumber:212222240033
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Placement_Data_Full_Class.csv")
df
df.head()
df.tail()
df.info()
df=df.drop('sl_no',axis=1)
df
df=df.drop(['ssc_b','hsc_b','gender'],axis=1)
df
df.shape
Data Encoding

df['degree_t']=df['degree_t'].astype('category')
df['workex']=df['workex'].astype('category')
df['specialisation']=df['specialisation'].astype('category')
df['status']=df['status'].astype('category')
df['hsc_s']=df['hsc_s'].astype('category')
df.info()
df['degree_t']=df['degree_t'].cat.codes
df['workex']=df['workex'].cat.codes
df['specialisation']=df['specialisation'].cat.codes
df['status']=df['status'].cat.codes
df['hsc_s']=df['hsc_s'].cat.codes
df.info()
df
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0]])
```

## Output:
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/5744a900-b2f5-4d08-8cd0-082735afe8dd)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/1ccee05e-6d77-4d80-9215-ea05a0aef39f)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/cc5e02f1-e514-4d89-b47d-55454df0332f)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/b22d4388-b20a-4325-ae65-24692a43f04d)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/fd9e0aa3-62c1-4719-9b03-7ffe8b66eb75)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/d418d031-f334-4181-8d1c-41770535a9b5)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/da4fd722-bab9-48cd-b6d4-2c9857ab1a7f)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/b10a97d8-5b83-4419-9fb7-1010a12eb611)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/ba563f72-267c-469e-a7ca-4e9bda279d31)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/66e209e9-874f-4849-ab82-61719ff6fbb4)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/cd489ab8-18ba-4072-93cb-3b7e7f572d01)
![image](https://github.com/R-Guruprasad/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390308/54a60e55-32f5-4c6d-992d-760ccf9409bf)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
