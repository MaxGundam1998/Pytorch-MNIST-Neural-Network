import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt



class MNDataset(Dataset):
    def __init__ (self, csv_file):
        self.data = pd.read_csv(csv_file, header=None)
        #Extract Label data of the first column
        self.labels = torch.LongTensor(self.data.iloc[:, 0].values)
        #Skip data of first column, and grab all the values
        self.features = torch.FloatTensor(self.data.iloc[:, 1:].values)

    def __len__ (self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        feature = self.features[idx]
        return (feature, label)
    
class NeuralNet(nn.Module):
    #Creating Neural Network with one Hidden Layer
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        #no activation and no softmax at the end
        return out


#Start of coding
input_size = 784
hidden_size = 128
num_classes = 10
num_epochs = 10
batch_size = 100
learn_rate = 0.001

#Make sure cuda is running
if not torch.cuda.is_available():
    raise Exception("CUDA is not available, cannot run on CPU.")

device = torch.device('cuda')

#Grabbed dataset
train_dataset = MNDataset('MNData/mnist_train.csv')
test_dataset = MNDataset('MNData/mnist_test.csv')

train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Create the model
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


#Training Phase
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #Orign shape: [100, 1, 28, 28]
        #Resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #Forward Propagation
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Print the loss every 100th step
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
print('Start Testing')

with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value, index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the Network on the test images: {acc} %')
