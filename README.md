# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import and Load Data:
Import necessary libraries (pandas, sklearn) and load the employee churn dataset.

Preprocess Data:
Handle missing values, encode categorical variables, and separate features (X) and target (y).

Split Data:
Divide the dataset into training and testing sets using train_test_split().

Train Model:
Create and train a DecisionTreeClassifier on the training data.

Predict and Evaluate:
Use the model to predict churn on test data and evaluate performance using accuracy and confusion matrix.

## Program:
```
import pandas as pd
import numpy as np
data= pd.read_csv('Employee.csv')
data.head()
```
<img width="1052" height="194" alt="image" src="https://github.com/user-attachments/assets/89e3be65-4d34-4b0f-88b7-41e49ad120a0" />

```
data.info()
```
<img width="512" height="322" alt="image" src="https://github.com/user-attachments/assets/241ac591-b627-45ab-87df-1a2e6ec81cdb" />

```
data.isnull().sum()
```
<img width="328" height="208" alt="image" src="https://github.com/user-attachments/assets/6c87a6f1-b3ca-48a5-9ff0-7080655a1094" />

```
data["left"].value_counts()
```
<img width="287" height="69" alt="image" src="https://github.com/user-attachments/assets/eeaa2850-37d6-42b7-a583-b5d278b517e6" />

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
<img width="1061" height="190" alt="image" src="https://github.com/user-attachments/assets/b0580e51-50d2-418c-80e5-0859c784ed06" />

```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
<img width="967" height="168" alt="image" src="https://github.com/user-attachments/assets/04f2e04b-3440-4083-929d-5f772b4f3e36" />

```
y=data[["left"]]
y.head()
```
<img width="113" height="175" alt="image" src="https://github.com/user-attachments/assets/fe9648fc-047d-488b-9451-9e62b2b98765" />

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred= dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
confusion=metrics.confusion_matrix(y_test,y_pred)
classification=metrics.classification_report(y_test,y_pred)
print("Accuracy:")
print(accuracy)
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
print("Name: NIVETHA S")
print("Reg no: 212223040137")
print(y_pred)

```

## Output:
<img width="1919" height="1014" alt="image" src="https://github.com/user-attachments/assets/6478b0cc-7ed0-4b01-8e2d-d1f5b323a91f" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
