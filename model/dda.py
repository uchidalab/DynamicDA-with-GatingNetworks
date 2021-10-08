# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast
from torchinfo import summary

batch_size = 256
data_ch = 1
data_len = 1008
z_size = 512
n_classes = 7

data_len_after_cnn = int((((((data_len-2)/2)-2)/2)-2)/2)

class TSEncoder(nn.Module): # CNN + MLP, Time Series Encoder
    # initializers
    def __init__(self, data_len_after_cnn, d=32):
        super().__init__()
        self.data_len_after_cnn = data_len_after_cnn
        self.conv1 = nn.Conv1d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm1d(128)

        self.linear1 = nn.Linear(128*self.data_len_after_cnn, 512)
        self.linear_bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, d)
        self.linear_bn2 = nn.BatchNorm1d(d)

        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool1d(2, 2)

    # forward method
    def forward(self, x):
        x = self.maxpooling(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpooling(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpooling(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.linear_bn1(self.linear1(x)))
        x = self.relu(self.linear_bn2(self.linear2(x)))

        return x

#data = torch.ones((batch_size, data_ch, data_len))
#ts_encoder = TSEncoder(data_len_after_cnn)
#print(ts_encoder(data).size())
#summary(ts_encoder, (batch_size,data_ch,data_len))

class Classifier(nn.Module): # MLP
    # initializers
    def __init__(self, num_of_class, d=32):
        super().__init__()
        self.num_of_class = num_of_class
        self.linear1 = nn.Linear(d, 512)
        self.linear_bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, self.num_of_class)

        self.relu = nn.ReLU()

    # forward method
    def forward(self, x):
        x = self.relu(self.linear_bn1(self.linear1(x)))
        x = self.linear2(x)

        return x

#z = torch.ones((batch_size, z_size))
#classifier = Classifier(n_classes, z_size)
#print(classifier(z).size())
#summary(classifier, (batch_size,z_size))

class Gating(nn.Module): # CNN + MLP
    # initializers
    def __init__(self, data_len_after_cnn, d=32):
        super().__init__()
        self.data_len_after_cnn = data_len_after_cnn
        self.conv1 = nn.Conv1d(1*2, 32, 3, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.linear1 = nn.Linear(128*self.data_len_after_cnn, 512)
        self.linear_bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 512)
        self.linear_bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 1)
        
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool1d(2, 2)
        self.sigmoid = nn.Sigmoid()

    # forward method
    def forward(self, x):
        x = self.maxpooling(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpooling(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpooling(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)

        x = self.relu(self.linear_bn1(self.linear1(x)))
        x = self.relu(self.linear_bn2(self.linear2(x)))
        x = self.sigmoid(self.linear3(x))
        return x.view(-1)

#data = torch.ones((batch_size, data_ch*2, data_len))
#gating = Gating(data_len_after_cnn)
#print(gating(data).size())
#summary(gating, (batch_size,data_ch*2,data_len))

class Gating5(nn.Module): # CNN + MLP
    # initializers
    def __init__(self, data_len_after_cnn, d=32):
        super().__init__()
        self.data_len_after_cnn = data_len_after_cnn
        self.conv1 = nn.Conv1d(1*5, 32, 3, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.linear1 = nn.Linear(128*self.data_len_after_cnn, 512)
        self.linear_bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 512)
        self.linear_bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 5)
        
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool1d(2, 2)
        self.softmax = nn.Softmax(dim=1)

    # forward method
    def forward(self, x):
        x = self.maxpooling(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpooling(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpooling(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)

        x = self.relu(self.linear_bn1(self.linear1(x)))
        x = self.relu(self.linear_bn2(self.linear2(x)))
        x = self.softmax(self.linear3(x)) # sum will be 1.0
        
        return x.view(-1,5)

#data = torch.ones((batch_size, data_ch*5, data_len))
#gating5 = Gating5(data_len_after_cnn)
#print(gating5(data).size())
#summary(gating5, (batch_size,data_ch*5,data_len))
