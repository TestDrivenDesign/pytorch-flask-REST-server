import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import io
from PIL import Image


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def get_net():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 128, 3)
            self.conv3 = nn.Conv2d(128, 128, 3)
            self.fc1 = nn.Linear(int(8192/16), 1024)
            self.fc2 = nn.Linear(1024, 256)
            self.fc3 = nn.Linear(256, 7)
    
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))   
            x = x.view(-1, int(8192/16))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    PATH = './test_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH, map_location='cpu'), strict=False)
    net.eval()

    return net

def get_img_tensor(img_bytes):
    transform = transforms.Compose([
        transforms.Resize(1023),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dimensions = (32, 32)
    # img = cv2.imread(img_path)
    # img = cv2.resize(img, dimensions)
    img =Image.open(io.BytesIO(img_bytes))
    img_tensor = transform(img)
    return img_tensor 
