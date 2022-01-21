import os
import torch
import torch.nn as nn
import functools
from collections import OrderedDict

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
    def __init__(self, device, input_nc, output_nc, lr, beta1, isTrain):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
        """
        super(Pix2PixModel2, self).__init__()
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.debug = False
        self.batch_norm = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        self.isTrain = isTrain
        self.device = device
        self.real_A = None
        self.real_B = None
        self.image_paths = None
        
        self.netG = self.define_G(input_nc, output_nc, ngf=64)
        self.lambda_L1 = 1.0

        if self.isTrain:
            self.netD = self.define_D(input_nc + output_nc, ndf=64)

            # define loss functions
            self.criterionGAN = nn.BCEWithLogitsLoss()
            self.criterionL1 = torch.nn.L1Loss()
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.register_buffer('real_label', torch.tensor(1.0))
            self.register_buffer('fake_label', torch.tensor(0.0))

            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

    def define_D(self, input_nc, ndf=64, n_layers=3):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        use_bias = False
        for n in range(1, n_layers): # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), 
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] # output 1 channel prediction map
        return nn.Sequential(*sequence)

    def define_G(self, input_nc, output_nc, ngf):
        use_bias = False
        self.netG = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i # 1, 2
            self.netG += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.BatchNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        n_blocks = 3
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            self.netG += [
                ResnetBlock(ngf * mult)
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i) # 4, 2
            self.netG += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                nn.BatchNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        
        self.netG += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        return nn.Sequential(*self.netG)

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        if self.debug:
            print("forward")
            print("\tself.real_A:", self.real_A.shape)
            print("\tself.fake_B:", self.fake_B.shape)

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        target_fake = self.fake_label.expand_as(pred_fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, target_fake)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        target_real = self.real_label.expand_as(pred_real)
        self.loss_D_real = self.criterionGAN(pred_real, target_real)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        if self.debug:
            print("backward_D")
            print("\tfake_AB:", fake_AB.shape)
            print("\tpred_fake:", pred_fake.shape)
            print("\tself.loss_D_fake:", self.loss_D_fake.shape)
            print("# Real")
            print("\treal_AB:", real_AB.shape)
            print("\tpred_real:", pred_real.shape)
            print("\tself.loss_D_real:", self.loss_D_real.shape)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        target_real = self.real_label.expand_as(pred_fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, target_real)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

        if self.debug:
            print("backward_G")
            print("\tfake_AB:", fake_AB.shape)
            print("\tself.loss_G_GAN:", self.loss_G_GAN.shape)
            print("\tself.loss_G_L1:", self.loss_G_L1.shape)
            print("\tself.loss_G:", self.loss_G.shape)
    

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']
    
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, save_path):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                torch.save(net.state_dict(), save_path)
                    
def create_model(device, opt):
    
    return Pix2PixModel2(device, opt.input_nc, opt.output_nc, opt.lr, opt.beta1, opt.isTrain).to(device)