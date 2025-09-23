import torch
from torch import nn
from sympy import factorint
from torch.nn.functional import unfold

class Down(nn.Module):
    def __init__(self, channels, sampling):
        super(Down, self).__init__()
        self.sampling = sampling
        conv_layers = []
        decimation_layers = []

        for p, exp in factorint(sampling).items():
            kernel = p+1 if p %2 == 0 else p+2
            for _ in range(0, exp):
                conv = nn.Conv2d(in_channels=channels,out_channels=channels,
                                             kernel_size=kernel,
                                             padding=kernel // 2,
                                             bias=False,
                                             groups=channels
                                            # OJO AMB EL GROUPS = channels
                                             )
                with torch.no_grad():
                    conv.weight.zero_()
                    center = conv.kernel_size[0] // 2  # Asumimos kernel cuadrado.
                    for i in range(channels):
                        conv.weight[i, 0, center, center] = 1.0
                conv_layers.append(conv)
                decimation_layers.append(PatchAvg(p))

        self.conv_k = nn.ModuleList(conv_layers)
        self.decimation = nn.ModuleList(decimation_layers)

    def forward(self, input):
        list = [input]
        for i, conv in enumerate(self.conv_k):
            input = conv(input)
            input = self.decimation[i](input)
            list.append(input)
        return input
class PatchAvg(nn.Module):
    def __init__(self, sampling):
        super(PatchAvg, self).__init__()
        self.sampling = sampling

    def forward(self, input):
        B, C, H, W = input.size()
        downsamp = torch.zeros(B, C, H//self.sampling, W//self.sampling, device=input.device)
        for i in range(self.sampling):
            for j in range(self.sampling):
                downsamp[:,:,:,:] += input[:,:,i::self.sampling,j::self.sampling] / ( self.sampling ** 2 )
        return downsamp

class Upsampling(nn.Module):
    def __init__(self, channels, sampling):
        super(Upsampling, self).__init__()
        self.sampling = sampling
        up_layers = []
        for p, exp in factorint(sampling).items():
            for _ in range(exp):
                kernel = p + 1 if p % 2 == 0 else p + 2
                up_layers.append(nn.ConvTranspose2d(in_channels=channels,
                                                    out_channels=channels,
                                                    kernel_size=kernel,
                                                    stride=p,
                                                    padding=kernel // 2,
                                                    bias=False,
                                                    output_padding=p - 1,
                                                    groups=channels))
        self.up = nn.Sequential(*up_layers)
    def forward(self, input):
        return self.up(input)

class interpolate():
    def __init__(self, scale_factor, mode):
        self.scale_factor = scale_factor
        self.mode = mode
    def __call__(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
