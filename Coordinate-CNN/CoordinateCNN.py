import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Coordinate-Cnn",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "CNN",
    "dataset": "Custom",
    "epochs": 10,
    }
)

# Size of the image
size = (64, 64)

# List to hold your dataset
dataset = []
for i in range(size[0]):
    for j in range(size[1]):
        img = np.ones((size[0], size[1], 3), dtype=np.uint8)
        img[i][j] = [0, 0, 0]
        dataset.append((img, (i, j)))

# Split the dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

class DotDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        # Convert the image and label to PyTorch tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        label = torch.tensor(label).float() / (size[0] - 1)
        return image, label

class DotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the data loaders
train_dataloader = DataLoader(DotDataset(train_dataset), batch_size=32, shuffle=True)
val_dataloader = DataLoader(DotDataset(val_dataset), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DotNet().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
best_val_loss = float('inf')
for epoch in range(10):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Average Training Loss: {running_loss / len(train_dataloader)}')
    wandb.log({"Training Loss": running_loss / len(train_dataloader)})

    # Validation loop
    model.eval()
    with torch.no_grad():
        running_val_loss = 0.0
        for i, data in enumerate(val_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            running_val_loss += val_loss.item()
        average_val_loss = running_val_loss / len(val_dataloader)
        print(f'Epoch {epoch + 1}, Average Validation Loss: {average_val_loss}')
        wandb.log({"Validation Loss": average_val_loss})

        # Save model if it's the best one so far
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

print('Finished Training')

def display_predictions(model, dataloader, device, num_images=5):
    for i in range(12):
        next(iter(dataloader))
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    # Move images, labels, and outputs back to cpu for visualization
    images = images.cpu().numpy()
    labels = labels.cpu().numpy() * (size[0] - 1)  # rescale the labels
    outputs = outputs.cpu().numpy() * (size[0] - 1)  # rescale the outputs

    for i in range(num_images):
        plt.figure()
        img = images[i].transpose((1, 2, 0))  # convert to (Height, Width, Channel) for display
        plt.imshow(img)
        plt.title(f"True label: {labels[i]}, Predicted label: {outputs[i]}")
        plt.show()

# Test the function
display_predictions(model, train_dataloader, device)
