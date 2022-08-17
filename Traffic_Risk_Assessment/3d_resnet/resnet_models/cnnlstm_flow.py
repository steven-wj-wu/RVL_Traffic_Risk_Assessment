import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet50_ori, resnet101


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet50_ori(pretrained=True)
        #self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2)
        #self.gru = nn.GRU(input_size=2048, hidden_size=512, num_layers=2)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        

        """
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.conv1 = nn.Conv3d(
            3,
           64,
           kernel_size=7,
           stride=(1, 2, 2),
            padding=(3, 3, 3),
           bias=False)
        """
       
    def forward(self, x_3d):

        #x_3d = self.conv1(x_3d)
        #x_3d = self.bn1(x_3d)
        #x_3d = self.relu(x_3d)
        #x_3d = self.maxpool(x_3d)
        #print(x_3d.size(1)//5)
        #b = x_3d[:, 15:20, :, :, :]
        #print(b.size())
        #input()
        #print(x_3d.size())
        #input() 

        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)
            #out, hidden = self.gru(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x