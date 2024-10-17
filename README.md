# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Load the dataset and perform any necessary preprocessing, such as handling missing values
 and encoding categorical variables.
2. Initialize the logistic regression model and train it using the training data.
3. Use the trained model to predict the placement status for the test set
4. Evaluate the model using accuracy and confusion matrix
  
  
  

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Raja rithika 
RegisterNumber:2305001029  
*/
import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(["sl_no",'salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender']=le.fit_transform(data1['gender'])
data1['ssc_b']=le.fit_transform(data1['ssc_b'])
data1['hsc_b']=le.fit_transform(data1['hsc_b'])
data1['hsc_s']=le.fit_transform(data1['hsc_s'])
data1['degree_t']=le.fit_transform(data1['degree_t'])
data1['workex']=le.fit_transform(data1['workex'])
data1['specialisation']=le.fit_transform(data1['specialisation'])
data1['status']=le.fit_transform(data1['status'])
data1
x=data1.iloc[:,:-1]
x
y=data1.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred,x_test
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("accuracy score:",accuracy)
print("\nconfusion matrix:\n",confusion)
print("\nclassification report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion)
cm_display.plot()
```

## Output:
![image](https://github.com/user-attachments/assets/d1ff1fb5-2d86-4b78-84f7-c66b776e15c8)
![image](https://github.com/user-attachments/assets/10a4f26e-f96a-4c1f-a1ae-c879896f7ddd)
![image](https://github.com/user-attachments/assets/589452f1-70a3-438c-b716-d8db22a235a7)
![image](https://github.com/user-attachments/assets/52468c9a-c418-45f7-b9e0-19e56bae84b8)
![image](https://github.com/user-attachments/assets/070a0252-d461-4bac-81d6-b2830e7a1e2e)
![image](https://github.com/user-attachments/assets/f8cb2526-52e1-4592-be97-a0dcae8fdf15)
![image](https://github.com/user-attachments/assets/75caefe8-fb00-4b13-9850-81335309ce4b)
![image](https://github.com/user-attachments/assets/f5a08023-c8b1-4132-ba95-cf762afb46a5)
![image](https://github.com/user-attachments/assets/e05b91ab-b039-466b-b508-bbc80583ee27)
![image](https://github.com/user-attachments/assets/00a19eb6-d430-43bf-b385-51700bcf2ed3)













## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
