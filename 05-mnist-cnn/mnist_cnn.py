import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_data = datasets.MNIST(root = 'data', train= True, download= True, transform= transforms.ToTensor()) 
test_data = datasets.MNIST(root = 'data', train= False, download= True, transform= transforms.ToTensor()) 

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.Dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 3 * 3 , 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))    
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 16 * 3 * 3)
        x = self.Dropout(x)
        x = self.fc1(x) 

        return x    
    
model = CNN()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images , labels in train_loader:

        optimizer.zero_grad()
        train_predict = model(images)
        loss = loss_fn(train_predict , labels)
        loss.backward()
        optimizer.step()

        total_loss +=loss.item()

    average_loss = total_loss/len(train_loader)
    print(f"Epoch: {epoch+1}, average_loss: {average_loss}")

correct = 0
total=0
model.eval()
with torch.no_grad():

    for images, labels in test_loader:

        test_predict = model(images)
        test_predict = torch.argmax(test_predict, dim=1)
        correct +=  (test_predict == labels).sum().item()
        total += labels.size(0)

    accuracy = (correct/total)*100
    print(f"Accuracy: {accuracy:.4f}%")             
    