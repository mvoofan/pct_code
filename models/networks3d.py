import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import os

###############################################################################
# Functions
###############################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        #netG = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        netG = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_copy':
        netG = UnetCopy(norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        #netG.cuda(device_id=gpu_ids[0])
        netG.cuda(    device = gpu_ids[0] )  #modified
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        #netD.cuda(device_id=gpu_ids[0])
        netD.cuda(device=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

    
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

#############################
#
# 3D version of UnetGenerator
class UnetGenerator3d(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]): # TODO
        super(UnetGenerator3d, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True) 
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout) 
        unet_block = UnetSkipConnectionBlock3d(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer) 
        unet_block = UnetSkipConnectionBlock3d(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer) 
        unet_block = UnetSkipConnectionBlock3d(ngf, ngf * 2, unet_block, norm_layer=norm_layer) 
        unet_block = UnetSkipConnectionBlock3d(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
#revise kernel_size to 3 to ensure the image size are the same
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)

class UnetCopy(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetCopy, self).__init__()
        self.gpu_ids = gpu_ids
        self.norm_layer=nn.BatchNorm3d(256)
        self.relu=nn.LeakyReLU(0.2, True)
        self.use_bias= True
        self.input_channel=2

        self.convert_input_to_bp_conv11=nn.Sequential(  self.downconv1(self.input_channel,32), nn.LeakyReLU(0.2, True) )
        self.convert_bp_conv11_to_bp_conv12=nn.Sequential( self.downconv1(32,32) , nn.LeakyReLU(0.2, True) )
        
        self.convert_bp_conv12_to_bp_conv21=nn.Sequential(  self.downconv2(32,64), nn.LeakyReLU(0.2, True) )
        self.convert_bp_conv21_to_bp_conv22=nn.Sequential( self.downconv1(64,64) , nn.LeakyReLU(0.2, True) )

        self.convert_bp_conv22_to_bp_conv31=nn.Sequential(  self.downconv2(64,128), nn.LeakyReLU(0.2, True) )
        self.convert_bp_conv31_to_bp_conv32=nn.Sequential(  self.downconv1(128,128), nn.LeakyReLU(0.2, True) )

        self.convert_bp_conv32_to_bp_conv41=nn.Sequential(  self.downconv2(128,256), nn.LeakyReLU(0.2, True) )
        self.convert_bp_conv41_to_bp_conv42=nn.Sequential(  self.downconv1(256,256), nn.LeakyReLU(0.2, True) )

        self.convert_bp_conv42_to_bp_deconv21=nn.Sequential(  nn.BatchNorm3d(256), self.upconv2(256,128), nn.LeakyReLU(0.2, True) )
        self.convert_bp_deconv21_to_bp_deconv22=nn.Sequential(  nn.BatchNorm3d(256), self.downconv1(256,128), nn.LeakyReLU(0.2, True) )

        self.convert_bp_deconv22_to_bp_deconv31=nn.Sequential(  nn.BatchNorm3d(128), self.upconv2(128,64), nn.LeakyReLU(0.2, True) )
        self.convert_bp_deconv31_to_bp_deconv32=nn.Sequential(  nn.BatchNorm3d(128), self.downconv1(128,64), nn.LeakyReLU(0.2, True) )

        self.convert_bp_deconv32_to_bp_deconv41=nn.Sequential(  nn.BatchNorm3d(64), self.upconv2(64,32), nn.LeakyReLU(0.2, True) )
        self.convert_bp_deconv41_to_bp_deconv42=nn.Sequential(  nn.BatchNorm3d(64), self.downconv1(64,32), nn.LeakyReLU(0.2, True) )

        self.convert_bp_deconv42_tobp_deconv51=nn.Sequential(  nn.BatchNorm3d(32), self.downconv1(32,1), nn.LeakyReLU(0.2, True) )
        
        self.convet_bp_deconv52_to_bp_Cn1=nn.Sequential(  nn.BatchNorm3d(1+self.input_channel), self.downconv1(1+self.input_channel,32), nn.LeakyReLU(0.2, True) )
        self.convert_bp_Cn1_to_bp_Mo1cn1=nn.Sequential(  nn.BatchNorm3d(32), self.downconv1(32,32), nn.LeakyReLU(0.2, True) )
        self.convert_bp_Mo1cn1_to_bp_Mo1cn2=nn.Sequential(  nn.BatchNorm3d(32), self.downconv1(32,32), nn.LeakyReLU(0.2, True) )

        self.convert_bp_Mo6_to_bp_Cn7=nn.Sequential(  nn.BatchNorm3d(64), self.downconv1(64,32), nn.LeakyReLU(0.2, True) )
        #self.convert_bp_Cn7_to_out=nn.Sequential(  nn.BatchNorm3d(32), self.downconv1(32,1), nn.LeakyReLU(0.2, True) )
        self.convert_bp_Cn7_to_out=nn.Sequential(  nn.BatchNorm3d(32), self.downconv1(32,1), nn.Tanh() )



        






    

    def downconv1(self, in_channel, output_channel):
        return nn.Conv3d(in_channel , output_channel, kernel_size=3, stride=1, padding=1, bias=True)
    def downconv2(self, in_channel, output_channel):
        return nn.Conv3d(in_channel , output_channel, kernel_size=4, stride=2, padding=1, bias=True)

    def upconv1(self, in_channel, output_channel):
        return nn.ConvTranspose3d(in_channel , output_channel, kernel_size=3, stride=1, padding=1, bias=True)
    def upconv2(self, in_channel, output_channel):
        return nn.ConvTranspose3d(in_channel , output_channel, kernel_size=4, stride=2, padding=1, bias=True)
    

    def pass_fun(self, input):
        norm_layer=nn.BatchNorm3d
        use_bias= True

        
        
        bp_conv11 = nn.Sequential(  downconv1(2,32), nn.LeakyReLU(0.2, True) )(input)
        bp_conv12 = nn.Sequential( downconv1(32,32) , nn.LeakyReLU(0.2, True) )(bp_conv11)

        bp_conv21 = nn.Sequential(  downconv2(32,64), nn.LeakyReLU(0.2, True) )(bp_conv12)
        bp_conv22 = nn.Sequential( downconv1(64,64) , nn.LeakyReLU(0.2, True))(bp_conv21)

        bp_conv31 = nn.Sequential(  downconv2(64,128), nn.LeakyReLU(0.2, True) )(bp_conv22)
        bp_conv32 = nn.Sequential(  downconv1(128,128), nn.LeakyReLU(0.2, True) )(bp_conv31)

        bp_conv41 = nn.Sequential(  downconv2(128,156), nn.LeakyReLU(0.2, True) )(bp_conv32)
        bp_conv42 = nn.Sequential(  downconv1(256,256), nn.LeakyReLU(0.2, True) )(bp_conv41)


        bp_deconv21_gt=nn.Sequential(  norm_layer, upconv2(256,128), nn.LeakyReLU(0.2, True) )(bp_conv42)
        bp_deconv21_gt=nn.Sequential(  nn.LeakyReLU(0.2, True) )( torch.cat( [bp_conv32, bp_deconv21_gt], 1))
        bp_deconv22_gt=nn.Sequential(  norm_layer, downconv1(256,128), nn.LeakyReLU(0.2, True) )(bp_deconv21_gt)

        bp_deconv31_gt=nn.Sequential(  norm_layer, upconv2(128,64), nn.LeakyReLU(0.2, True) )(bp_deconv22_gt)
        bp_deconv31_gt=nn.Sequential(  nn.LeakyReLU(0.2, True) )( torch.cat( [bp_conv22, bp_deconv31_gt], 1))
        bp_deconv32_gt=nn.Sequential(  norm_layer, downconv1(128,64), nn.LeakyReLU(0.2, True) )(bp_deconv31_gt)

        bp_deconv41_gt=nn.Sequential(  norm_layer, upconv2(64,32), nn.LeakyReLU(0.2, True) )(bp_deconv32_gt)
        bp_deconv41_gt=nn.Sequential(  nn.LeakyReLU(0.2, True) )( torch.cat( [bp_conv12, bp_deconv41_gt], 1))
        bp_deconv42_gt=nn.Sequential(  norm_layer, downconv1(64,32), nn.LeakyReLU(0.2, True) )(bp_deconv41_gt)

        bp_deconv51_gt=nn.Sequential(  norm_layer, downconv1(32,1), nn.LeakyReLU(0.2, True) )(bp_deconv42_gt)
        bp_deconv52_gt=nn.Sequential(  nn.LeakyReLU(0.2, True) )( torch.cat( [bp_deconv51_gt, input], 1))
        
        bp_Cn1_gt=nn.Sequential(  norm_layer, downconv1(2,32), nn.LeakyReLU(0.2, True) )(bp_deconv52_gt)
        bp_Mo1cn1_gt=nn.Sequential(  norm_layer, downconv1(32,32), nn.LeakyReLU(0.2, True) )(bp_Cn1_gt)
        bp_Mo1cn2_gt=nn.Sequential(  norm_layer, downconv1(32,32), nn.LeakyReLU(0.2, True) )(bp_Mo1cn1_gt)

        bp_Mo1re4_gt = bp_Cn1_gt + bp_Mo1cn2_gt

        bp_Mo1re4_gt = nn.Sequential( nn.LeakyReLU(0.2, True) ) (bp_Mo1re4_gt)

        bp_Mo6_gt = torch.cat([bp_Cn1_gt,bp_Mo1re4_gt], 1)

        bp_Cn7_gt = nn.Sequential(  norm_layer, downconv1(64,32), nn.LeakyReLU(0.2, True) )(bp_Mo6_gt)

        bp2im_out_gt=nn.Sequential(  norm_layer, downconv1(32,1), nn.LeakyReLU(0.2, True) )(bp_Cn7_gt)

        return bp2im_out_gt

    
    def forward(self, input):


        '''
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            #return self.pass_fun(input)
        '''
        bp_conv11= self.convert_input_to_bp_conv11(input)
        bp_conv12= self.convert_bp_conv11_to_bp_conv12(bp_conv11)

        bp_conv21=self.convert_bp_conv12_to_bp_conv21(bp_conv12)
        bp_conv22=self.convert_bp_conv21_to_bp_conv22(bp_conv21)

        bp_conv31=self.convert_bp_conv22_to_bp_conv31(bp_conv22)
        bp_conv32=self.convert_bp_conv31_to_bp_conv32(bp_conv31)

        bp_conv41=self.convert_bp_conv32_to_bp_conv41(bp_conv32)
        bp_conv42=self.convert_bp_conv41_to_bp_conv42(bp_conv41)

        bp_deconv21_gt=self.convert_bp_conv42_to_bp_deconv21(bp_conv42)
        bp_deconv21_gt=self.relu( torch.cat( [bp_conv32, bp_deconv21_gt], 1))
        bp_deconv22_gt=self.convert_bp_deconv21_to_bp_deconv22(bp_deconv21_gt)

        bp_deconv31_gt=self.convert_bp_deconv22_to_bp_deconv31(bp_deconv22_gt)
        bp_deconv31_gt=self.relu( torch.cat( [bp_conv22, bp_deconv31_gt], 1))
        bp_deconv32_gt=self.convert_bp_deconv31_to_bp_deconv32(bp_deconv31_gt)

        bp_deconv41_gt=self.convert_bp_deconv32_to_bp_deconv41(bp_deconv32_gt)
        bp_deconv41_gt=self.relu( torch.cat( [bp_conv12, bp_deconv41_gt], 1))
        bp_deconv42_gt=self.convert_bp_deconv41_to_bp_deconv42(bp_deconv41_gt)

        bp_deconv51_gt=self.convert_bp_deconv42_tobp_deconv51(bp_deconv42_gt)
        bp_deconv52_gt=self.relu( torch.cat( [bp_deconv51_gt, input], 1))

        bp_Cn1_gt=self.convet_bp_deconv52_to_bp_Cn1(bp_deconv52_gt)
        bp_Mo1cn1_gt=self.convert_bp_Cn1_to_bp_Mo1cn1(bp_Cn1_gt)
        bp_Mo1cn2_gt=self.convert_bp_Mo1cn1_to_bp_Mo1cn2(bp_Mo1cn1_gt)

        bp_Mo1re4_gt = bp_Cn1_gt + bp_Mo1cn2_gt

        bp_Mo1re4_gt=self.relu(bp_Mo1re4_gt)
        bp_Mo6_gt = torch.cat([bp_Cn1_gt,bp_Mo1re4_gt], 1)

        bp_Cn7_gt=self.convert_bp_Mo6_to_bp_Cn7(bp_Mo6_gt)
        bp2im_out_gt=self.convert_bp_Cn7_to_out(bp_Cn7_gt)

        return bp2im_out_gt





        


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d 
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)