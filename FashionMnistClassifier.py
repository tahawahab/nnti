import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

class NeuralNetworkClassifier(nn.Module):
    def __init__(self):
        super(NeuralNetworkClassifier, self).__init__()
        self.hl1 = nn.Linear(28*28, 128)
        self.hl2 = nn.Linear(128, 64)
        self.ol = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.hl1(x))
        x = F.relu(self.hl2(x))
        x = self.ol(x)
        return x

def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0], data[1]
            outputs = model(images.view(-1, 28*28))
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    return 100 * correct / total

ds = FashionMNIST(root='./data', train=True, download=True, transform=ToTensor())
train = int(0.8 * len(ds))
val = int(0.1 * len(ds))
test = len(ds) - train - val
train_ds, val_ds, test_ds = random_split(ds, [train, val, test])

train_load = DataLoader(train_ds, batch_size=64, shuffle=True)
val_load = DataLoader(val_ds, batch_size=64, shuffle=False)
test_load = DataLoader(test_ds, batch_size=64, shuffle=False)

model = NeuralNetworkClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    loop_loss = 0.0
    for i, data in enumerate(train_load, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loop_loss = loop_loss + loss.item()

    print(f'Epoch {epoch + 1}, Loss: {loop_loss / len(train_load)}')

train_accuracy = calculate_accuracy(train_load, model)
val_accuracy = calculate_accuracy(val_load, model)
test_accuracy = calculate_accuracy(test_load, model)

print(f'Training Accuracy: {train_accuracy}%')
print(f'Validation Accuracy: {val_accuracy}%')
print(f'Test Accuracy: {test_accuracy}%')
