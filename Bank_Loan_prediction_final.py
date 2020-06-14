import tkinter as tk 
from tkinter import messagebox,simpledialog,filedialog
from tkinter import *
import tkinter
from imutils import paths
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

root= tk.Tk() 
root.title("Bank Loan Prediction")
root.geometry("1300x1200")

global data_train, data_test
def upload_data():
    global train_data
    train_data = askopenfilename(initialdir = "Dataset")
    #pathlabel.config(text=train_data)
    text.insert(END,"Dataset loaded\n\n")

def upload_new():
    global test_data
    text.delete('1.0',END)
    test_data = askopenfilename(initialdir = "Dataset")
    #pathlabel1.config(text=test_data)
    text.insert(END,"New Dataset loaded\n\n")

def data():
    text.delete('1.0',END)
    data_train = pd.read_csv(train_data)
    data_test = pd.read_csv(test_data)
    text.insert(END,"Top FIVE rows of the Dataset\n\n")
    text.insert(END,data_train.head())
    text.insert(END,"\n\nTop FIVE rows of the New Dataset\n\n")
    text.insert(END,data_test.head())
    return data_train,data_test

def statistics():
    text.delete('1.0',END)
    global data_train,data_test
    data_train = pd.read_csv(train_data)
    data_test = pd.read_csv(test_data)
    text.insert(END,"Top FIVE rows of the Dataset\n\n")
    text.insert(END,data_train.head())
    text.insert(END,"\n\nTop FIVE rows of the New Dataset\n\n")
    text.insert(END,data_test.head())
    train_stats=data_train.describe()
    test_stats=data_test.describe()
    text.insert(END,"\n\nStatistical Measurements for Data\n\n")
    text.insert(END,train_stats)
    text.insert(END,"\n\nStatistical Measurements for New Data\n\n")
    text.insert(END,test_stats)
    null=data_train.isnull().sum()
    G_V=data_train['Gender'].value_counts()
    M_V=data_train['Married'].value_counts()
    SE_V=data_train['Self_Employed'].value_counts()
    CH_V=data_train['Credit_History'].value_counts()
    text.insert(END,"\n\nDisplaying Number of Missing Values in all Independent Attributes\n\n")
    text.insert(END,null)
    text.insert(END,"\n\nDisplaying Value counts for Major Attributes like Gender, Married, Self_Employed, Credit_History Attributes respectively\n\n")
    text.insert(END,G_V)
    text.insert(END,"\n")
    text.insert(END,M_V)
    text.insert(END,"\n")
    text.insert(END,SE_V)
    text.insert(END,"\n")
    text.insert(END,CH_V)


def Preprocess():
    global combined,data_train,data_test
    text.delete('1.0',END)
    text.insert(END,"\t\t\tHandled Missing values based on Value_Counts\n")
    text.insert(END,"\t\t\t\t\t&\n")
    text.insert(END,"\t\tEncoded Categorical Variables and Feature Scaling for High Magnitude variables\n\n")
    train = data_train
    test = data_test
    targets = train['Loan_Status']
    train.drop('Loan_Status', 1, inplace=True)
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'Loan_ID'], inplace=True, axis=1)
    combined['Gender'].fillna('Male', inplace=True)
    combined['Married'].fillna('Yes', inplace=True)
    combined['Self_Employed'].fillna('No', inplace=True)
    combined['LoanAmount'].fillna(combined['LoanAmount'].median(), inplace=True)
    combined['Credit_History'].fillna(2, inplace=True) #For unknown credit card history consider 2

    combined['Gender'] = combined['Gender'].map({'Male':1,'Female':0})
    combined['Married'] = combined['Married'].map({'Yes':1,'No':0})
    combined['Singleton'] = combined['Dependents'].map(lambda d: 1 if d=='1' else 0)
    combined['Small_Family'] = combined['Dependents'].map(lambda d: 1 if d=='2' else 0)
    combined['Large_Family'] = combined['Dependents'].map(lambda d: 1 if d=='3+' else 0)
    combined.drop(['Dependents'], axis=1, inplace=True)
    combined['Education'] = combined['Education'].map({'Graduate':1,'Not Graduate':0})
    combined['Self_Employed'] = combined['Self_Employed'].map({'Yes':1,'No':0})
    combined['Total_Income'] = combined['ApplicantIncome'] + combined['CoapplicantIncome']
    combined.drop(['ApplicantIncome','CoapplicantIncome'], axis=1, inplace=True)
    combined['Debt_Income_Ratio'] = combined['Total_Income'] / combined['LoanAmount']
    combined['Very_Short_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t<=60 else 0)
    combined['Short_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>60 and t<180 else 0)
    combined['Long_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>=180 and t<=300  else 0)
    combined['Very_Long_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>300 else 0)
    combined.drop('Loan_Amount_Term', axis=1, inplace=True)
    combined['Credit_History_Bad'] = combined['Credit_History'].map(lambda c: 1 if c==0 else 0)
    combined['Credit_History_Good'] = combined['Credit_History'].map(lambda c: 1 if c==1 else 0)
    combined['Credit_History_Unknown'] = combined['Credit_History'].map(lambda c: 1 if c==2 else 0)
    combined.drop('Credit_History', axis=1, inplace=True)
    property_dummies = pd.get_dummies(combined['Property_Area'], prefix='Property')
    combined = pd.concat([combined, property_dummies], axis=1)
    combined.drop('Property_Area', axis=1, inplace=True)
    sc=MinMaxScaler()
    combined['LoanAmount']=sc.fit_transform(combined[['LoanAmount']])
    combined['Total_Income']=sc.fit_transform(combined[['Total_Income']])
    combined['Debt_Income_Ratio']=sc.fit_transform(combined[['Debt_Income_Ratio']])
    text.insert(END,"\nChecking Missing Values\n")
    text.insert(END,combined.isnull().sum())
    text.insert(END,"\n\nDisplaying Top FIVE rows of Preprocessed data\n\n")
    text.insert(END,combined.head())
    return combined

def train_test():
    global train_data,combined,New_data
    global x_train,x_test,y_train,y_test
    text.delete('1.0',END)
    Loan = pd.read_csv(train_data)
    targets = Loan['Loan_Status'].map({'Y':1,'N':0})
    train = combined.head(614)
    New_data = combined.iloc[614:]
    x_train,x_test,y_train,y_test=train_test_split(train,targets,test_size=0.2,random_state=25)
    text.insert(END,"Train and Test model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(train))+"\n")
    text.insert(END,"Training Size : "+str(len(x_train))+"\n")
    text.insert(END,"Test Size : "+str(len(x_test))+"\n")
    return x_train,x_test,y_train,y_test


# Machine Learning Models

def RF():
    global New_data,data_test
    global x_train,x_test,y_train,y_test
    global new_x_train,new_x_test,new_data
    text.delete('1.0',END)
    text.insert(END,"\t\t\t\tRandom Forest Classifier\n\n")
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(x_train,y_train)
    features = pd.DataFrame()
    features['Feature'] = x_train.columns
    features['Importance'] = clf.feature_importances_
    features.sort_values(by=['Importance'], ascending=False, inplace=True)
    features.set_index('Feature', inplace=True)
    text.insert(END,"Selected Important Features Automatically by using *feature_importances_* & *SelectFromModel*\n\n")
    text.insert(END,features[:5])
    selector = SelectFromModel(clf, prefit=True)
    train_reduced = selector.transform(x_train)
    new_x_train=pd.DataFrame(train_reduced,columns=['Debt_Income_Ratio','Credit_History_Bad','Total_Income','LoanAmount','Credit_History_Good'])
    test_reduced = selector.transform(x_test)
    new_x_test=pd.DataFrame(test_reduced,columns=['Debt_Income_Ratio','Credit_History_Bad','Total_Income','LoanAmount','Credit_History_Good'])
    new_reduced=selector.transform(New_data)
    new_data=pd.DataFrame(new_reduced,columns=['Debt_Income_Ratio','Credit_History_Bad','Total_Income','LoanAmount','Credit_History_Good'])
    parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}

    rf = RandomForestClassifier(**parameters)
    rf.fit(new_x_train, y_train)
    pred=rf.predict(new_x_test)
    acc=accuracy_score(y_test,pred)
    cm=confusion_matrix(y_test,pred)
    CR=classification_report(y_test,pred)
    output = rf.predict(new_data).astype(int)
    df_output = pd.DataFrame()
    df_output['Loan_ID'] = data_test['Loan_ID']
    df_output['Loan_Predicted_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output)
    df_output[['Loan_ID','Loan_Predicted_Status']].to_csv('Loan_Approval_Prediction_submission@RF.csv',index=False)
    text.insert(END,"\n\nConfusion Matrix:\n"+str(cm)+"\n\n")
    text.insert(END,"Accuracy Score:\n"+str(np.round(acc*100,4))+' %'+"\n\n")
    text.insert(END,"Predicted Values on Test Data:\n"+str(pred)+"\n\n")
    text.insert(END,"Classification Report:\n"+str(CR))
    text.insert(END,"\n\nFinal Predicted values on New Data:\n\n")
    text.insert(END,df_output)
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")
    

def LR():
    global new_data,data_test
    global x_train,x_test,y_train,y_test
    global new_x_train,new_x_test,new_data
    text.delete('1.0',END)
    text.insert(END,"\t\t\t\tLogistic Regression\n\n")
    model=LogisticRegression()
    model.fit(new_x_train,y_train)
    pred=model.predict(new_x_test)
    acc=accuracy_score(y_test,pred)
    cm =confusion_matrix(y_test,pred)
    CR =classification_report(y_test,pred)
    text.insert(END,"\n\nConfusion Matrix:\n"+str(cm)+"\n\n")
    text.insert(END,"Accuracy:\n"+str(np.round(acc*100,4))+' %'+"\n\n")
    text.insert(END,"Predicted Values on Test Data:\n"+str(pred)+"\n\n")
    text.insert(END,"Classification Report:\n"+str(CR))
    output = model.predict(new_data).astype(int)
    df_output = pd.DataFrame()
    df_output['Loan_ID'] = data_test['Loan_ID']
    df_output['Loan_Predicted_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output)
    df_output[['Loan_ID','Loan_Predicted_Status']].to_csv('Loan_Approval_Prediction_submission@LR.csv',index=False)
    text.insert(END,"\n\nFinal Predicted values on New Data:\n\n")
    text.insert(END,df_output)
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")

def KNN():
    global new_data,data_test
    global x_train,x_test,y_train,y_test
    global new_x_train,new_x_test,new_data
    text.delete('1.0',END)
    text.insert(END,"\t\t\t\tK-Nearest Neighbors Classifier\n\n")
    model=KNeighborsClassifier()
    model.fit(new_x_train,y_train)
    pred=model.predict(new_x_test)
    acc=accuracy_score(y_test,pred)
    cm =confusion_matrix(y_test,pred)
    CR =classification_report(y_test,pred)
    text.insert(END,"\n\nConfusion Matrix:\n"+str(cm)+"\n\n")
    text.insert(END,"Accuracy:\n"+str(np.round(acc*100,4))+' %'+"\n\n")
    text.insert(END,"Predicted Values on Test Data:\n"+str(pred)+"\n\n")
    text.insert(END,"Classification Report:\n"+str(CR))
    output = model.predict(new_data).astype(int)
    df_output = pd.DataFrame()
    df_output['Loan_ID'] = data_test['Loan_ID']
    df_output['Loan_Predicted_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output)
    df_output[['Loan_ID','Loan_Predicted_Status']].to_csv('Loan_Approval_Prediction_submission@KNN.csv',index=False)
    text.insert(END,"\n\nFinal Predicted values on New Data:\n\n")
    text.insert(END,df_output)
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")

def SVM():
    global new_data,data_test
    global x_train,x_test,y_train,y_test
    global new_x_train,new_x_test,new_data
    text.delete('1.0',END)
    text.insert(END,"\t\t\t\tSupport Vector Classifier\n\n")
    model=SVC()
    model.fit(new_x_train,y_train)
    pred=model.predict(new_x_test)
    acc=accuracy_score(y_test,pred)
    cm =confusion_matrix(y_test,pred)
    CR =classification_report(y_test,pred)
    text.insert(END,"\n\nConfusion Matrix:\n"+str(cm)+"\n\n")
    text.insert(END,"Accuracy:\n"+str(np.round(acc*100,4))+' %'+"\n\n")
    text.insert(END,"Predicted Values on Test Data:\n"+str(pred)+"\n\n")
    text.insert(END,"Classification Report:\n"+str(CR))
    output = model.predict(new_data).astype(int)
    df_output = pd.DataFrame()
    df_output['Loan_ID'] = data_test['Loan_ID']
    df_output['Loan_Predicted_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output)
    df_output[['Loan_ID','Loan_Predicted_Status']].to_csv('Loan_Approval_Prediction_submission@SVM.csv',index=False)
    text.insert(END,"\n\nFinal Predicted values on New Data:\n\n")
    text.insert(END,df_output)
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")

def DT():
    global new_data,data_test
    global x_train,x_test,y_train,y_test
    global new_x_train,new_x_test,new_data
    text.delete('1.0',END)
    text.insert(END,"\t\t\t\tDecision Tree Classifier\n\n")
    model=DecisionTreeClassifier()
    model.fit(new_x_train,y_train)
    pred=model.predict(new_x_test)
    acc=accuracy_score(y_test,pred)
    cm =confusion_matrix(y_test,pred)
    CR =classification_report(y_test,pred)
    text.insert(END,"\n\nConfusion Matrix:\n"+str(cm)+"\n\n")
    text.insert(END,"Accuracy:\n"+str(np.round(acc*100,4))+' %'+"\n\n")
    text.insert(END,"Predicted Values on Test Data:\n"+str(pred)+"\n\n")
    text.insert(END,"Classification Report:\n"+str(CR))
    output = model.predict(new_data).astype(int)
    df_output = pd.DataFrame()
    df_output['Loan_ID'] = data_test['Loan_ID']
    df_output['Loan_Predicted_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output)
    df_output[['Loan_ID','Loan_Predicted_Status']].to_csv('Loan_Approval_Prediction_submission@DT.csv',index=False)
    text.insert(END,"\n\nFinal Predicted values on New Data:\n\n")
    text.insert(END,df_output)
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")


def input_values():
    text.delete('1.0',END)
    global New_data,data_test
    global x_train,x_test,y_train,y_test
    global new_x_train,new_x_test,new_data
    global rf
    global Debt_Income_Ratio #our 1st input variable
    Debt_Income_Ratio = float(entry1.get()) 
    
    global Total_Income #our 2nd input variable
    Total_Income = float(entry2.get())

    global Credit_History_Bad 
    Credit_History_Bad = float(entry3.get())

    global LoanAmount
    LoanAmount = float(entry4.get())

    global Credit_History_Good
    Credit_History_Good = float(entry5.get())
    
    list1=[[Debt_Income_Ratio,Total_Income,Credit_History_Bad,LoanAmount,Credit_History_Good]]

    parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}

    rf = RandomForestClassifier(**parameters)
    rf.fit(new_x_train, y_train)

    
    Prediction_result  = rf.predict(list1)
    text.insert(END,"Among all Classifiers Random Forest Classifier having greater accuracy score\n\n")
    text.insert(END,"New values are predicted from Random Forest Classifier\n\n")
    text.insert(END,"Predicted Loan Status for the New inputs\n\n")
    text.insert(END,np.vectorize(lambda s: 'Yes' if s==1 else 'No')(Prediction_result))

font = ('times', 14, 'bold')
title = Label(root, text='Bank Loan Prediction Using Machine Learning')  
title.config(font=font)           
title.config(height=2, width=120)       
title.place(x=0,y=5)

font1 = ('times',13 ,'bold')
button1 = tk.Button (root, text='Upload Data',width=13,command=upload_data) 
button1.config(font=font1)
button1.place(x=60,y=100)

button2 = tk.Button (root, text='Upload New Data',width=13,command=upload_new)
button2.config(font=font1)
button2.place(x=60,y=150)

button3 = tk.Button (root, text='Statistics',width=13,command=statistics)  
button3.config(font=font1)
button3.place(x=60,y=200)

button3 = tk.Button (root, text='Preprocessing',width=13,command=Preprocess)
button3.config(font=font1) 
button3.place(x=60,y=250)

button4 = tk.Button (root, text='Train & Test',width=13,command=train_test)
button4.config(font=font1) 
button4.place(x=60,y=300)

title = Label(root, text='Application of ML models')
#title.config(bg='RoyalBlue2', fg='white')  
title.config(font=font1)           
title.config(width=25)       
title.place(x=250,y=70)

button5 = tk.Button (root, text='Random Forest',width=15,bg='pale green',command=RF)
button5.config(font=font1) 
button5.place(x=300,y=100)

button6 = tk.Button (root, text='KNN',width=15,bg='sky blue',command=KNN)
button6.config(font=font1) 
button6.place(x=300,y=150)

button7 = tk.Button (root, text='SVM',width=15,bg='orange',command=SVM)
button7.config(font=font1) 
button7.place(x=300,y=200)

button7 = tk.Button (root, text='Decision Tree',width=15,bg='violet',command=DT)
button7.config(font=font1) 
button7.place(x=300,y=250)

button7 = tk.Button (root, text='Logistic Regression',width=15,bg='indian red',command=LR)
button7.config(font=font1) 
button7.place(x=300,y=300)

title = Label(root, text='Enter Input values for the New Prediction')
title.config(bg='black', fg='white')  
title.config(font=font1)           
title.config(width=40)       
title.place(x=60,y=380)

font3=('times',9,'bold')
title1 = Label(root, text='*You Should enter scaled values between 0 and 1')
 
title1.config(font=font3)           
title1.config(width=40)       
title1.place(x=50,y=415)

def clear1(event):
    entry1.delete(0, tk.END)

font2=('times',10)
entry1 = tk.Entry (root) # create 1st entry box
entry1.config(font=font2)
entry1.place(x=60, y=450,height=30,width=150)
entry1.insert(0,'Debt_Income_Ratio')
entry1.bind("<FocusIn>",clear1)

def clear2(event):
    entry2.delete(0, tk.END)

font2=('times',10)
entry2 = tk.Entry (root) # create 1st entry box
entry2.config(font=font2)
entry2.place(x=315, y=450,height=30,width=150)
entry2.insert(0,'Total_Income')
entry2.bind("<FocusIn>",clear2)


def clear3(event):
    entry3.delete(0, tk.END)

font2=('times',10)
entry3 = tk.Entry (root) # create 1st entry box
entry3.config(font=font2)
entry3.place(x=60, y=500,height=30,width=150)
entry3.insert(0,'Credit_History_Bad')
entry3.bind("<FocusIn>",clear3)

def clear4(event):
    entry4.delete(0, tk.END)

font2=('times',10)
entry4 = tk.Entry (root) # create 1st entry box
entry4.config(font=font2)
entry4.place(x=315, y=500,height=30,width=150)
entry4.insert(0,'LoanAmount')
entry4.bind("<FocusIn>",clear4)

def clear5(event):
    entry5.delete(0, tk.END)

font2=('times',10)
entry5 = tk.Entry (root) # create 1st entry box
entry5.config(font=font2)
entry5.place(x=60, y=550,height=30,width=150)
entry5.insert(0,'Credit_History_Good')
entry5.bind("<FocusIn>",clear5)





Prediction = tk.Button (root, text='Prediction',width=15,fg='white',bg='green',command=input_values)
Prediction.config(font=font1) 
Prediction.place(x=180,y=650)



font1 = ('times', 11, 'bold')
text=Text(root,height=32,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set,xscrollcommand=scroll.set)
text.place(x=550,y=70)
text.config(font=font1)

#root.config(bg='grey')
root.mainloop()