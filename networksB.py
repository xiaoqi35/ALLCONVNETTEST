from torch.autograd import Variable
import torch.nn as nn
from torch import mean
from torch.nn import init
import torch.nn.functional as F
import math
class NetB(nn.Module):
    def __init__(self):
        super(NetB,self).__init__()
        
        self.conv1 = nn.Conv2d(3,96,5)
        self.conv2 = nn.Conv2d(96,96,1)
        self.pool1 = nn.MaxPool2d(3,2)
        
        self.conv3 = nn.Conv2d(96,192,5)
        self.conv4 = nn.Conv2d(192,192,1)
        self.pool2 = nn.MaxPool2d(3,2)
        
        self.conv5 = nn.Conv2d(192,192,3)
        self.conv6 = nn.Conv2d(192,192,1)
        self.conv7 = nn.Conv2d(192,10,1)
        
       
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))              
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))                  
        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
    
        x = mean(x,2)
        x = mean(x,2)
       
        x = F.softmax(x,1)
        return x
netB= NetB()

