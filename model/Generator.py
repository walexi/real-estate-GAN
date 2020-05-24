import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #  Convolutional layers 
        
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1, padding = 2)
        self.conv1_bn = nn.BatchNorm2d(16)
        
        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv2_bn = nn.BatchNorm2d(32)
        
        # input 265x256x32  output 128x128x64
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.conv3_bn = nn.BatchNorm2d(64)
        
        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Conv2d(64, 128, 5, stride = 2, padding = 2)
        self.conv4_bn = nn.BatchNorm2d(128)
        
        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Conv2d(128, 128, 5, stride = 2, padding = 2)
        self.conv5_bn = nn.BatchNorm2d(128)
        
        # convs for global features
        # input 32x32x128 output 16x16x128
        self.conv51 = nn.Conv2d(128,128,5, stride =2 , padding =2 )
        
        # input 16x16x128 output 8x8x128
        self.conv52 = nn.Conv2d(128,128,5, stride =2 , padding =2 )
        
        # input 8x8x128 output 1x1x128
        self.conv531 = nn.Conv2d(128,128,5, stride =2 , padding =1 )
        
        # input 1x1x128 output 1x1x128
        self.conv532 = nn.Conv2d(128,128,5, stride =2 , padding =1 )
        
        # input 32x32x128 output 32x32x128
        # the global features should be concatenated to the feature map aftere this layer
        # the output after concat would be 32x32x256
        self.conv6 = nn.Conv2d(128, 128, 5, stride = 1, padding =2)
        
        # input 32x32x256 output 32x32x128
        self.conv7 = nn.Conv2d(256, 128, 5, stride = 1, padding = 2)
        
        # deconvolutional layers
        # input 32x32x128 output 64x64x128
        self.dconv1 = nn.ConvTranspose2d(128, 128, 4, stride = 2, padding = 1)
        self.dconv1_bn = nn.BatchNorm2d(128)
        
        # input 64x64x256 ouput 128x128x128
        self.dconv2 = nn.ConvTranspose2d(256, 128, 4, stride = 2, padding = 1)
        self.dconv2_bn = nn.BatchNorm2d(256)
        
        # input 128x128x192 output 256x256x64
        self.dconv3 = nn.ConvTranspose2d(192, 64, 4, stride = 2, padding = 1)
        self.dconv3_bn = nn.BatchNorm2d(192)
        
        # input 256x256x96 ouput 512x512x32
        self.dconv4 = nn.ConvTranspose2d(96, 32, 4, stride = 2, padding = 1)
        self.dconv4_bn = nn.BatchNorm2d(96)
        
        # final convolutional layers
        # input 512x512x48 output 512x512x16
        self.conv8 = nn.Conv2d(48, 16, 5, stride = 1, padding = 2)
        self.conv8_bn = nn.BatchNorm2d(48)
        
        # input 512x512x16 output 512x512x3
        self.conv9 = nn.Conv2d(16, 3, 5, stride = 1, padding = 2)    
        self.conv9_bn = nn.BatchNorm2d(16)
        # SELU
                
    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x = self.conv1_bn(F.selu(self.conv1(x)))

        # input 512x512x16 to output 256x256x32
        x1 = self.conv2_bn(F.selu(self.conv2(x)))

        # input 256x256x32 to output 128x128x64
        x2 = self.conv3_bn(F.selu(self.conv3(x1)))

        # input 128x128x64 to output 64x64x128
        x3 = self.conv4_bn(F.selu(self.conv4(x2)))

        # input 64x64x128 to output 32x32x128
        x4 = self.conv5_bn(F.selu(self.conv5(x3)))

        #convolutions for global features
        # input 32x32x128 to output 16x16x128
        x51 = self.conv51(x4)
        
        # input 16x16x128 to output 8x8x128
        x52 = self.conv52(x51)
        
        # input 8x8x128 to output 1x1x128
        x53 = self.conv532(F.selu(self.conv531(x52)))
        x53_temp = torch.cat([x53]*32,dim = 2 )
        x53_temp = torch.cat([x53_temp]*32,dim=3)

        # input 32x32x256 to output 32x32x128
        x5 = self.conv6(x4)
        
        # input 32x32x128 to output 32x32x128
        x5 = self.conv7(torch.cat([x5,x53_temp],dim=1))

        # input 32x32x128 to output 64x64x128
        xd = self.dconv1(self.dconv1_bn(F.selu(x5)))

        # input 64x64x256 to output 128x128x128
        xd = self.dconv2(self.dconv2_bn(F.selu(torch.cat([xd,x3], dim=1))))

        # input 128x128x192 to output 256x256x64
        xd = self.dconv3(self.dconv3_bn(F.selu(torch.cat([xd,x2],dim=1))))

        # input 256x256x64 to output 512x512x32
        xd = self.dconv4(self.dconv4_bn(F.selu(torch.cat([xd,x1],dim=1))))

        # input 512x512x48 to output 512x512x16
        xd = self.conv8(self.conv8_bn(F.selu(torch.cat([xd,x],dim=1))))

        # input 512x512x16 to output 512x512x3
        xd = self.conv9(self.conv9_bn(F.selu((xd))))
        return xd