from ucimlrepo import fetch_ucirepo     
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score


  
#  fetch dataset 
student_performance = fetch_ucirepo(id=320) 

  
# data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets['G3']
y = (y>=10).astype(int)
X=pd.get_dummies(X)

#print("___________________________Classical Models___________________________________________")  
#print(X.shape) 
#print(student_performance.variables) 

# X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y )
# model = DecisionTreeClassifier(max_depth=10,random_state=42)
# model.fit(X_train, y_train)

# y_predict = model.predict(X_test)
# print("Accuracy: ",accuracy_score(y_test,y_predict )) # with out stratify Accuracy: 87.6% , with stratify Accuracy : 77%
# print("\nClassification Report:\n")
# print(classification_report(y_test,y_predict))
  
# ------------------------------------------------------------------------------------------------------------------------------


X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y )
model1 = KNeighborsClassifier(n_neighbors=5)
model1.fit(X_train, y_train)

y_predict = model1.predict(X_test)
scores = cross_val_score(model1 , X, y, cv=5)
print("Scores" ,scores)
print("Average: ",scores.mean())
print("Accuracy: ",accuracy_score(y_predict, y_test )) # Accuracy: 83%


# ------------------------------------------------------------------------------------------------------------------------------
  

# X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y )
# model2 = RandomForestClassifier(n_estimators = 100)
# model2.fit(X_train, y_train)

# y_predict = model2.predict(X_test)
# print("Accuracy: ",accuracy_score(y_predict, y_test )) # with stratify Accuracy: 82%, with out stratify Accuracy: 86.9%


#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# print(f"________________________Deep Learning________________________________________")

# def PrepareData(X,y):
#     X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y )
#     X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32 )
#     X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32 )
#     y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32 ).unsqueeze(1)
#     y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32 ).unsqueeze(1)

#     return  X_train,  X_test, y_train, y_test

# class StudentPerformanceModel(nn.Module):
#     def __init__(self,input_features):
#         super().__init__()
#         self.layer1 = nn.Linear(input_features, 16)
#         self.layer2 = nn.Linear(16,1)

#     def forward(self, x):
#         x= torch.relu(self.layer1(x))    
#         x= torch.sigmoid(self.layer2(x))

#         return x

# def train(model, X_train, y_train, loss_fn, optimizer):
#     model.train()
#     epochs =100

#     for epoch in range(epochs):

#         predict = model(X_train)
#         loss = loss_fn(predict, y_train )
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if epoch%10==0:
#             print(f"Epoch: {epoch}, loss: {loss.item():.4f}")

# def evaluate(model, X_test,y_test):
#     model.eval()
#     with torch.no_grad():

#         y_predict = model(X_test)
#         y_predict = (y_predict>=0.5).float()
#         accuracy = (y_predict==y_test).float().mean()
#         print("Accuracy: ",accuracy.item())  #Accuracy 84.6%

# if __name__ == "__main__":

#     X=X.astype('float32')

#     X_train,  X_test, y_train, y_test=PrepareData(X,y)
#     model = StudentPerformanceModel(X_train.shape[1])
#     loss_fn = nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     train(model, X_train, y_train, loss_fn, optimizer)
#     evaluate(model, X_test,y_test)




    