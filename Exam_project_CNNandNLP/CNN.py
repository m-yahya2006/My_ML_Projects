from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader 
import torch.nn as nn 

train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.ToTensor()) 
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transforms.ToTensor())

#image , label = train_data[0]
#print(image.shape)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3 , padding=1 )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3 , padding=1 )
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3 , padding=1 )
        self.Droupout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32 * 4 * 4 , 10)

    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1 , 32 * 4 * 4)
        x= self.Droupout(x)
        x= self.fc1(x)
        return x

model = CNN()
loss_fn= nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters() ,lr = 0.01)

epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images , labels in train_dataloader:
        optim.zero_grad()
        output = model(images)
        loss = loss_fn(output , labels)
        loss.backward()
        optim.step()

        total_loss+=loss
    print(f"epoch: {epoch}, Loss: {total_loss/len(train_dataloader)}")


with torch.no_grad():
    correct = 0
    total = 0
    model.eval()
    for images , labels in test_dataloader:
        output = model(images)
        ourput = torch.argmax(output , dim=1)

        correct += (output==labels).sum().item()
        total += labels.size(0)

    print(f"Accuracy: {correct/total}")




    

