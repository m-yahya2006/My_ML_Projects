import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

df = pd.read_csv(url, header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

X = df[['variance', 'skewness', 'curtosis', 'entropy']]
y= df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train= torch.tensor(X_train.values, dtype=torch.float32)
y_train= torch.tensor(y_train.values, dtype=torch.float32)
X_test= torch.tensor(X_test.values, dtype=torch.float32)
y_test= torch.tensor(y_test.values, dtype=torch.float32)

class BankNote(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1= nn.Linear(4, 8)
        self.layer2= nn.Linear(8, 1)

    def forward(self, x):
        x= torch.relu(self.layer1(x))
        x= torch.sigmoid(self.layer2(x))

        return x    
    
model = BankNote()

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.01) 

epochs= 100

for epoch in range(epochs):

    prediction = model(X_train)

    loss= loss_fn(prediction, y_train.unsqueeze(1))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 10 ==0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

with torch.no_grad():
    test_predict = model(X_test)
    test_predict = (test_predict>=0.5).float()

    accuracy = (test_predict.squeeze() == y_test).float().mean()

    print(f"Accuracy: {accuracy.item()}")        