from ucimlrepo import fetch_ucirepo  

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn



def PrepareData(X,y):
    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = torch.tensor(X_train, dtype=torch.float32 )
    X_test = torch.tensor(X_test, dtype=torch.float32 )
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32 ).unsqueeze(1)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32 ).unsqueeze(1)

    return  X_train,  X_test, y_train, y_test

class StudentPerformanceModel(nn.Module):
     def __init__(self,input_features):
         super().__init__()
         self.layer1 = nn.Linear(input_features, 16)
         self.layer2 = nn.Linear(16,1)

     def forward(self, x):
         x= torch.relu(self.layer1(x))    
         x= torch.sigmoid(self.layer2(x))

         return x
def train(model, X_train, y_train, loss_fn, optimizer):
     model.train()
     epochs =100

     for epoch in range(epochs):

         predict = model(X_train)
         loss = loss_fn(predict, y_train )
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         if epoch%10==0:
             print(f"Epoch: {epoch}, loss: {loss.item():.4f}")

def evaluate(model, X_test,y_test):
     model.eval()
     with torch.no_grad():

         y_predict = model(X_test)
         y_predict = (y_predict>=0.5).float()
         accuracy = (y_predict==y_test).float().mean()
         print("Accuracy: ",accuracy.item())  #Accuracy 84.6%


def main():
    #fetch dataset 
    student_performance = fetch_ucirepo(id=320) 

  
    #data (as pandas dataframes) 
    X = student_performance.data.features 
    X=pd.get_dummies(X)     
    y = student_performance.data.targets['G3']
    y = (y>=10).astype(int)
            # 0 -> Fail
            # 1 -> Pass
    # dataset analysis
    print(X.shape)
    # handle missing values
    if X.isnull().sum().sum()>0:
        X=X.dropna()
        y=y.loc[X.index]

   
    
    print("___________________________Classical Models___________________________________________")  

    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)

    # --------------------------------------------------------------------

    print("DecisionTree: \n")
    model = DecisionTreeClassifier(max_depth=10,random_state=42)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print("Accuracy:\n",accuracy_score(y_test,y_predict )) # with out stratify Accuracy: 87.6% , with stratify Accuracy : 77%
    print("\nClassification Report:\n")
    print(classification_report(y_test,y_predict)) 

    # --------------------------------------------------------------------  

    print("RandomForestClassifier\n")
    model2 = RandomForestClassifier(n_estimators = 100)
    model2.fit(X_train, y_train)
    y_predict = model2.predict(X_test)
    print("Accuracy:\n",accuracy_score(y_predict, y_test )) # with stratify Accuracy: 82%, with out stratify Accuracy: 86.9% 
    print("\nClassification Report:\n")
    print(classification_report(y_test,y_predict)) 

    # --------------------------------------------------------------------  

    
    print("KNeighborsClassification\n")
    model1 = KNeighborsClassifier(n_neighbors=5)
    model1.fit(X_train, y_train)
    y_predict = model1.predict(X_test)
    scores = cross_val_score(model1 , X, y, cv=5)
    print("Scores: \n" ,scores)
    print("\nAverageScore:\n ",scores.mean())
    print("\nAccuracy:\n ",accuracy_score(y_predict, y_test )) # Accuracy: 83%
    print("\nClassification Report:\n")
    print(classification_report(y_test,y_predict))

    
    # --------------------------------------------------------------------  


    print(f"________________________Deep Learning________________________________________")

    X=X.astype('float32')
     #Standard scalling    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) 
    X_train,  X_test, y_train, y_test=PrepareData(X,y)
    model = StudentPerformanceModel(X_train.shape[1])
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, X_train, y_train, loss_fn, optimizer)
    evaluate(model, X_test,y_test)

if __name__ == "__main__":
    main()

    




    