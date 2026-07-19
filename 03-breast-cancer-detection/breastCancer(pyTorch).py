from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


data = load_breast_cancer()
#print("Keys: ",data.keys())
#print("Discription: ",data.DESCR)
#print("Data: ",data.data[:10])
#print("FeaturesLength: ",len(data.feature_names))

df = pd.DataFrame(data.data, columns=data.feature_names)
#print(df.head())
#print(df.shape)

X = data.data
y = data.target

def prepareData(X,y):
    scaler = StandardScaler()

    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size= 0.2, random_state=42, stratify=y)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train,  y_test
class BreastCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(30, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x):
        x= torch.relu(self.layer1(x))
        x= torch.relu(self.layer2(x))
        x= torch.sigmoid(self.layer3(x))

        return x
    




def train(model , X_train, y_train, loss_fn, optimizer ):
    epochs = 200
    for epoch in range(epochs):

        train_predict = model(X_train)
        loss = loss_fn(train_predict.squeeze() , y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, loss: {loss}")


def evaluate(model, X_test, y_test):
    with torch.no_grad():
        test_predict= model(X_test)
        test_predict= (test_predict>=0.5).float()

        accuracy = (test_predict.squeeze()==y_test).float().mean()
        print("Accuracy: ",accuracy)


if __name__=="__main__":

    X_train, X_test, y_train,  y_test = prepareData(X, y)
    model = BreastCancerModel()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.01 )

    train(model , X_train, y_train, loss_fn, optimizer )
    evaluate(model, X_test, y_test)

    
    
