import torch
import torch.nn as nn 
import torch.nn.functional as F 
import math 

from utilits import * 


# In order to load pre-trained AE weight, We have to implement TF's SAME padding mode by torch.
class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """
    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride`, i.e., the "SAME" padding mode in Tensorflow,
    so the number of columns to delete:
        pad = 2*padding - output_padding = kernel - stride
    We can solve the above equation and get:
        padding = ceil((kernel - stride)/2), and
        output_padding = 2*padding - (kernel - stride) which is either 1 or 0.
    But to get the same result with Tensorflow, we should delete values by ourselves instead of using padding and
    output_padding in ConvTranspose2d.
    To get there, we check the following conditions:
    If pad = kernel - stride is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad = kernel - stride is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves; or we can use ConvTranspose2d to delete `pad - 1` by setting `padding=(pad - 1) / 2` and `ouput_padding=0`
    and then delete the last row/column of the resulting tensor by ourselves.
    Here we implement the former case.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    And this module can only output a tensor with shape `stride * size_input`.
    A more flexible module can be found in `yaleb.py` which can output arbitrary size as specified.
    """

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        '''
        param:
            channels: a list containing all channels in the network.
            kernels:  a list containing all kernels in the network.
        '''
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.encoder.add_module('pad%d' % (i + 1), Conv2dSamePad(kernels[i], 2))
            self.encoder.add_module('conv%d' % (i + 1), nn.Conv2d(channels[i], channels[i+1],kernel_size=kernels[i], stride=2))
            self.encoder.add_module('relu%d' % (i + 1), nn.ReLU(True))

        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        self.decoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.decoder.add_module('deconv%d' % (i + 1), nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('pad%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relu%d' % i, nn.ReLU(True))

    def forward(self, x):
        hidden = self.encoder(x)
        y = self.decoder(hidden)
        return y

class AutoEncoder(nn.Module):
    def __init__(self, channels):
        '''
        param:
            channels: a list containing all channels in the network.
        '''
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.encoder.add_module('fc%d' % (i + 1), nn.Linear(channels[i], channels[i+1]))
            self.encoder.add_module('relu%d' % (i + 1), nn.ReLU(True))

        channels = list(reversed(channels))
        self.decoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.decoder.add_module('deconv%d' % (i + 1), nn.Linear(channels[i], channels[i + 1]))
            self.decoder.add_module('relu%d' % i, nn.ReLU(True))

    def forward(self, x):
        hidden = self.encoder(x)
        y = self.decoder(hidden)
        return y

class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad= True)

    def forward(self, x):
        y = torch.matmul(self.Coefficient, x)
        return y 

class DSCNet(nn.Module):
    def __init__(self, channels, num_sample):
        super(DSCNet, self).__init__()
        # self.pixelBlockList = pixelBlockList
        self.n = num_sample
        # self.ae = ConvAE(channels, kernels)
        self.ae = AutoEncoder(channels)
        self.self_expression = SelfExpression(self.n)
        
    def forward(self, x):
        z = self.ae.encoder(x)

        shape = z.shape
        z = z.view(self.n, -1)
        z_recon = self.self_expression(z)
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)
        return x, x_recon, z, z_recon

    def conv_bn_relu(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
    )

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        
        # reconstruction loss
        loss_recon = F.mse_loss(x_recon, x, reduction='sum')
        # regularization loss
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        # expressiveness loss
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        loss = loss_recon + loss_coef * weight_coef + loss_selfExp * weight_selfExp
        
        return loss 
