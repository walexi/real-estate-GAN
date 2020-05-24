import torch.nn as nn
import torch.nn.functional as F
import torch


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #  Convolutional layers 
        
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1, padding = 2)
        self.conv1_in = nn.InstanceNorm2d(16)
        
        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv2_in = nn.InstanceNorm2d(32)
        
        # input 265x256x32  output 128x128x64
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.conv3_in = nn.InstanceNorm2d(64)
        
        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Conv2d(64, 128, 5, stride = 2, padding = 2)
        self.conv4_in = nn.InstanceNorm2d(128)
        
        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Conv2d(128, 128, 5, stride = 2, padding = 2)
        self.conv5_in = nn.InstanceNorm2d(128)
        
        # input 32x32x128  output 16x16x128
        # the output of this layer we need layers for global features
        self.conv6 = nn.Conv2d(128, 128, 5, stride = 2, padding = 2)
        self.conv6_in = nn.InstanceNorm2d(128)
        
        # input 16x16x128  output 1x1x1
        # the output of this layer we need layers for global features
        self.conv7 = nn.Conv2d(128, 1, 16)
        self.conv7_in = nn.InstanceNorm2d(1)
        
    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x = self.conv1_in(F.leaky_relu(self.conv1(x)))

        # input 512x512x16 to output 256x256x32
        x = self.conv2_in(F.leaky_relu(self.conv2(x)))
        
        # input 256x256x32 to output 128x128x64
        x = self.conv3_in(F.leaky_relu(self.conv3(x)))
        
        # input 128x128x64 to output 64x64x128
        x = self.conv4_in(F.leaky_relu(self.conv4(x)))
        
        # input 64x64x128 to output 32x32x128
        x = self.conv5_in(F.leaky_relu(self.conv5(x)))
        
        # input 32x32x128 to output 16x16x128
        x = self.conv6_in(F.leaky_relu(self.conv6(x)))
        
        # input 16x16x128 to output 1x1x1
        x = self.conv7(x)
        x = F.leaky_relu(x)
        
        return x