import torch
import torch.nn as nn

def conv_block(inputs, outputs, kernel_size, stride=1):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel_size, stride=1),
                         nn.ReLU(),
                         nn.BatchNorm2d(outputs),
                         nn.MaxPool2d(2))

class CustomModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomModel, self).__init__()        
        
        self.conv1 = conv_block(in_channels, 32, 3)
        self.conv2 = conv_block(32, 64, 3)
        self.conv3 = conv_block(64, 128, 3)
        self.conv4 = conv_block(128, 256, 3)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.linear1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out_pool = self.global_avg_pool(out4)             # [B, 256, 1, 1]
        out_pool = out_pool.view(out_pool.size(0), -1)    # [B, 256]
        out5 = nn.functional.relu(self.linear1(out_pool)) 
        out5 = self.dropout(out5)
        final = self.linear2(out5)
        
        return final   