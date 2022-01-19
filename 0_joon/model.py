import torch

import torch.nn as nn
import functools

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.dim = dim
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        dim = self.dim
        conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.BatchNorm2d(dim)
        )
        return conv_block

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Pix2PixModel2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
        """
        super(Pix2PixModel2, self).__init__()
        self.batch_norm = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        
        use_bias = False
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i # 1, 2
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.BatchNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        n_blocks = 6
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [
                ResnetBlock(ngf * mult)
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i) # 4, 2
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                nn.BatchNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.resnet = nn.Sequential(*model)

    def forward(self, x):
        return self.resnet(x)

def create_model(opt):
    return Pix2PixModel2(opt.input_nc, opt.output_nc)