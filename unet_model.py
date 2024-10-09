import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        def block(in_channels, out_channels, kernel_size=3, padding=1, extra_layers=2):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                ]
            for _ in range(extra_layers):
                layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                ])

            return nn.Sequential(*layers)
    

        
        self.encoder1 = block(in_channels, 64, extra_layers=2)
        self.encoder2 = block(64, 128, extra_layers=2)
        self.encoder3 = block(128, 256, extra_layers=2)
        self.encoder4 = block(256, 512, extra_layers=2)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = block(1024, 512, extra_layers=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = block(512, 256, extra_layers=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = block(256, 128, extra_layers=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = block(128, 64, extra_layers=2)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        middle = self.middle(self.pool(e4))

        d4 = self.upconv4(middle)
        d4 = F.interpolate(d4, size=e4.size()[2:], mode='nearest') # nearest interpolation is better than bilinear, does not introduce new artifacts
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = F.interpolate(d3, size=e3.size()[2:], mode='nearest')
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = F.interpolate(d2, size=e2.size()[2:], mode='nearest')
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = F.interpolate(d1, size=e1.size()[2:], mode='nearest')
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)
        
        out = self.final(d1)
        
        return out
