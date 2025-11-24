import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention2(nn.Module):
    def __init__(self, u_channels, pan_channels, patch_size, window_size):
        super(MultiHeadAttention2, self).__init__()
        self.geometric_head = SelfAttention(u_channels=u_channels, pan_channels=8, patch_size=patch_size, window_size=window_size)
        self.spectral_head = SelfAttention(u_channels=u_channels, pan_channels=8, patch_size=1, window_size=window_size)
        self.mix_head = SelfAttention(u_channels=u_channels, pan_channels=16, patch_size=patch_size, window_size=window_size)
        self.mlp = nn.Linear(3, 1)
        self.proj_pan = nn.Conv2d(u_channels, 8, kernel_size=1)
        self.proj_u = nn.Conv2d(u_channels, 8, kernel_size=1)

    def forward(self, u, pan):
        pan_proj = self.proj_pan(pan)
        u_proj = self.proj_u(u)
        head1 = self.geometric_head(u, pan_proj)
        head2 = self.spectral_head(u, u_proj)
        head3 = self.mix_head(u, torch.concat([u_proj, pan_proj], dim=1))

        return self.mlp(torch.concat([head1, head2, head3], dim=4)).squeeze(4)

class MultiHeadAttention(nn.Module):
    def __init__(self, u_channels, pan_channels, patch_size, window_size):
        super(MultiHeadAttention, self).__init__()
        self.geometric_head = SelfAttention(u_channels=u_channels, pan_channels=pan_channels, patch_size=patch_size, window_size=window_size)
        self.spectral_head = SelfAttention(u_channels=u_channels, pan_channels=u_channels, patch_size=1, window_size=window_size)
        self.mix_head = SelfAttention(u_channels=u_channels, pan_channels=pan_channels + u_channels, patch_size=patch_size, window_size=window_size)
        self.mlp = nn.Linear(3, 1)

    def forward(self, u, pan):
        head1 = self.geometric_head(u, pan)
        head2 = self.spectral_head(u, u)
        head3 = self.mix_head(u, torch.concat([u, pan], dim=1))

        return self.mlp(torch.concat([head1, head2, head3], dim=4)).squeeze(4)


class SelfAttention(nn.Module):
    def __init__(self, u_channels, pan_channels, patch_size, window_size):
        super(SelfAttention, self).__init__()
        self.pan_channels = pan_channels
        self.u_channels = u_channels
        self.patch_size = patch_size
        self.window_size = window_size
        self.spatial_weights = weights(channels=pan_channels, window_size=window_size, patch_size=patch_size)
        self.g = nn.Conv2d(u_channels, u_channels, 1, bias=False)

    def forward(self, u, pan):
        b, c, h, w = u.size()
        weights = self.spatial_weights(pan)
        g = self.g(u)  # [b, 3, h, w]
        g = F.unfold(g, self.window_size, padding=self.window_size // 2)
        g = g.view(b, self.u_channels, self.window_size * self.window_size, -1)
        g = g.view(b, self.u_channels, self.window_size * self.window_size, h, w)
        g = g.permute(0, 3, 4, 2, 1)
        return torch.matmul(weights, g).permute(0, 4, 1, 2, 3)


class weights(torch.nn.Module):
    def __init__(self, channels,  window_size, patch_size):
        super(weights, self).__init__()
        self.channels = channels
        self.phi = nn.Conv2d(channels, channels, 1, bias=False)
        self.theta = nn.Conv2d(channels, channels, 1, bias=False)
        self.window_size = window_size
        self.patch_size = patch_size
        self.softmax = nn.Softmax(dim=-1)
        self.eps = 1e-6

    def forward(self, u):
        b, c, h, w = u.size()
        phi = self.phi(u)
        theta = self.theta(u)
        theta = F.unfold(theta, self.patch_size, padding=self.patch_size // 2)
        theta = theta.view(b, 1, c*self.patch_size * self.patch_size, -1)
        theta = theta.view(b, 1, c*self.patch_size * self.patch_size, h, w)
        theta = theta.permute(0, 3, 4, 1, 2)

        phi = F.unfold(phi, self.patch_size, padding=self.patch_size // 2)
        phi = phi.view(b, c * self.patch_size * self.patch_size, h, w)
        phi = F.unfold(phi, self.window_size, padding=self.window_size // 2)
        phi = phi.view(b, c * self.patch_size * self.patch_size, self.window_size * self.window_size, h, w)
        phi = phi.permute(0, 3, 4, 1, 2)

        att = torch.matmul(theta, phi)

        return self.softmax(att)

class ResBlock(nn.Module):
    def __init__(self,  kernel_size, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        features = self.conv1(x)
        features = self.relu(features)
        features = self.conv2(features)
        return self.relu(features + x)

class ResBlockBN(nn.Module):
    def __init__(self,  kernel_size, in_channels):
        super(ResBlockBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.InstanceNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.bn3 = nn.InstanceNorm2d(in_channels)


    def forward(self, x):
        features = self.conv1(x)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.conv2(features)
        features = self.bn2(features)
        features = self.relu(features)
        features = self.conv3(features)
        features = self.bn3(features)

        return self.relu(features + x)
class Res_NL2(torch.nn.Module):
    def __init__(self, in_channels, aux_channels, patch_size, window_size, kernel_size=3):
        super().__init__()

        self.Nonlocal = MultiHeadAttention2(u_channels=in_channels, pan_channels=aux_channels, patch_size=patch_size,
                                           window_size=window_size)
        self.mix = nn.Conv2d(in_channels=3 * in_channels, out_channels=in_channels, kernel_size=1)
        self.residual = nn.Sequential(*[ResBlock(kernel_size=kernel_size, in_channels=in_channels) for _ in range(3)])

    def forward(self, u, pan):
        # Multi Attention Component
        u_multi_att = self.Nonlocal(u, pan)
        u_aux = torch.cat([u_multi_att, u, pan], dim=1)
        u_aux = self.mix(u_aux)
        res = self.residual(u_aux)
        return res

class Res_NL(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, features_channels, patch_size, window_size, kernel_size=3):
        super().__init__()

        self.features_channels = features_channels

        # Nonlocal layers
        self.NL_feat_u = nn.Conv2d(in_channels=u_channels, out_channels=5, kernel_size=kernel_size, stride=1,
                                   bias=False, padding=kernel_size // 2)
        self.NL_feat_pan = nn.Conv2d(in_channels=pan_channels, out_channels=3, kernel_size=kernel_size, stride=1,
                                     bias=False, padding=kernel_size // 2)
        self.Nonlocal = MultiHeadAttention(u_channels=5, pan_channels=3, patch_size=patch_size,
                                           window_size=window_size)

        # Residual layers
        self.res_feat_u = nn.Conv2d(in_channels=u_channels, out_channels=features_channels,
                                    kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2)
        self.res_feat_pan = nn.Conv2d(in_channels=pan_channels, out_channels=5, kernel_size=kernel_size, stride=1,
                                        bias=False, padding=kernel_size // 2)
        self.res_recon = nn.Conv2d(in_channels=features_channels + 5 + 5, out_channels=u_channels,
                                   kernel_size=kernel_size,
                                   stride=1, bias=False, padding=kernel_size // 2)
        self.residual = nn.Sequential(*[ResBlock(kernel_size=kernel_size, in_channels=features_channels + 5 + 5) for _ in range(3)])

    def forward(self, u, pan):
        # Multi Attention Component
        u_features = self.NL_feat_u(u)
        pan_features = self.NL_feat_pan(pan)
        u_multi_att = self.Nonlocal(u_features, pan_features)

        # Residual Component
        u_features = self.res_feat_u(u)
        pan_features = self.res_feat_pan(pan)
        u_aux = torch.cat([u_multi_att, u_features, pan_features], dim=1)
        res = self.residual(u_aux)
        return self.res_recon(res)

class Res_NLv2(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, features_channels, patch_size, window_size, kernel_size=3):
        super().__init__()

        self.features_channels = features_channels

        # Nonlocal layers
        self.NL_feat_u = nn.Conv2d(in_channels=u_channels, out_channels=5, kernel_size=kernel_size, stride=1,
                                   bias=False, padding=kernel_size // 2)
        self.NL_feat_pan = nn.Conv2d(in_channels=pan_channels, out_channels=18, kernel_size=kernel_size, stride=1,
                                     bias=False, padding=kernel_size // 2)
        self.Nonlocal = MultiHeadAttention(u_channels=5, pan_channels=18, patch_size=patch_size,
                                           window_size=window_size)

        # Residual layers
        self.res_feat_u = nn.Conv2d(in_channels=u_channels, out_channels=features_channels,
                                    kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size // 2)
        self.res_feat_pan = nn.Conv2d(in_channels=pan_channels, out_channels=30, kernel_size=kernel_size, stride=1,
                                        bias=False, padding=kernel_size // 2)
        self.res_recon = nn.Conv2d(in_channels=features_channels + 5 + 30, out_channels=u_channels,
                                   kernel_size=kernel_size,
                                   stride=1, bias=False, padding=kernel_size // 2)
        self.residual = nn.Sequential(*[ResBlock(kernel_size=kernel_size, in_channels=features_channels + 5 + 30) for _ in range(3)])

    def forward(self, u, pan):
        # Multi Attention Component
        u_features = self.NL_feat_u(u)
        pan_features = self.NL_feat_pan(pan)
        u_multi_att = self.Nonlocal(u_features, pan_features)

        # Residual Component
        u_features = self.res_feat_u(u)
        pan_features = self.res_feat_pan(pan)
        u_aux = torch.cat([u_multi_att, u_features, pan_features], dim=1)
        res = self.residual(u_aux)
        return self.res_recon(res)
class ConvRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
class Res_Local(torch.nn.Module):
    def __init__(self, u_channels, pan_channels, features_channels,kernel_size=3, **kwargs):
        super(Res_Local, self).__init__()
        layers = []
        layers.append(ConvRelu(in_channels=u_channels, out_channels=features_channels, kernel_size=kernel_size))
        for _ in range(3):
            layers.append(ResBlock(in_channels=features_channels, kernel_size=kernel_size))
        layers.append(nn.Conv2d(in_channels=features_channels, out_channels=u_channels, kernel_size=kernel_size,
                                padding=kernel_size // 2))
        self.ResNet = nn.Sequential(*layers)

    def forward(self, error, pan):
        x = error
        return self.ResNet(x) + error