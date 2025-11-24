import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.fft import fft2 as fft2
from torch.fft import ifft2 as ifft2
# from pytorch_wavelets import DWTForward  # 2D Discrete Wavelet Transform

EPSILON = 1e-8
class MSE(torch.nn.Module):
    def __init__(self, ):
        super(MSE, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, output, gt):
        pred = output['pred']
        mse = self.MSE(pred, gt)
        return mse
class FullResFineTune(torch.nn.Module):
    def __init__(self, alpha):
        super(FullResFineTune, self).__init__()
        self.L1 = torch.nn.L1Loss()
        self.alpha=alpha

    def forward(self, output, gt):
        DBu = output['DBu']
        Brovey = output['Brovey']
        DBu_loss = self.L1(DBu, torch.zeros_like(DBu))
        Brovey_loss = self.L1(Brovey, torch.zeros_like(Brovey))
        return DBu_loss + self.alpha * Brovey_loss

class DBumlpFineTune(torch.nn.Module):
    def __init__(self, alpha):
        super(DBumlpFineTune, self).__init__()
        self.L1 = torch.nn.L1Loss()
        self.alpha=alpha
        self.MSE = torch.nn.MSELoss()


    def forward(self, output, gt):
        DBu = output['DBu']
        utilla = output['utilla']
        DBu_loss = self.L1(DBu, torch.zeros_like(DBu))
        geom_loss = self.MSE(utilla, torch.zeros_like(utilla))
        return DBu_loss + self.alpha * geom_loss

class DBuFineTune(torch.nn.Module):
    def __init__(self, alpha):
        super(DBuFineTune, self).__init__()
        self.L1 = torch.nn.L1Loss()
        self.alpha=alpha

    def forward(self, output, gt):
        DBu = output['DBu']
        DBu_loss = self.L1(DBu, torch.zeros_like(DBu))
        return DBu_loss

class BroveyFineTune(torch.nn.Module):
    def __init__(self, alpha):
        super(BroveyFineTune, self).__init__()
        self.L1 = torch.nn.L1Loss()
        self.alpha=alpha

    def forward(self, output, gt):
        Brovey = output['Brovey']
        Brovey_loss = self.L1(Brovey, torch.zeros_like(Brovey))
        return Brovey_loss

class RMSE(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(RMSE, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps  # para evitar sqrt(0)

    def forward(self, output, gt):
        pred = output['pred']
        return torch.sqrt(self.mse(pred, gt) + self.eps)


class L1(torch.nn.Module):
    def __init__(self, ):
        super(L1, self).__init__()
        self.L1 = torch.nn.L1Loss()

    def forward(self, output, gt):
        pred = output['pred']
        l1 = self.L1(pred, gt)
        return l1


class MSE_specific_bands(torch.nn.Module):
    def __init__(self, ):
        super(MSE_specific_bands, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.bands = [2,3,18,24,25,50,51, 33,45,55]

    def forward(self, output, gt):
        pred = output['pred']
        l1 = self.MSE(pred, gt)
        l1_bands = self.MSE(pred[:,self.bands,:,:], gt[:,self.bands,:,:])
        return l1+l1_bands
class L1_specific_bands(torch.nn.Module):
    def __init__(self, ):
        super(L1_specific_bands, self).__init__()
        self.L1 = torch.nn.L1Loss()
        self.bands = [2,3,18,24,25,50,51, 33,45,55]

    def forward(self, output, gt):
        pred = output['pred']
        l1 = self.L1(pred, gt)
        l1_bands = self.L1(pred[:,self.bands,:,:], gt[:,self.bands,:,:])
        return l1+l1_bands


class L1_ndvi(torch.nn.Module):
    def __init__(self, ):
        super(L1_ndvi, self).__init__()
        self.L1 = torch.nn.L1Loss()

    def forward(self, output, gt):
        pred = output['pred']
        l1 = self.L1(pred, gt)
        l1_ndvi = self.L1(self._ndvi(pred), self._ndvi(gt))
        l1_ndvi_mean = self.L1(self._ndvi_mean(pred), self._ndvi_mean(gt))
        # print("l1: ", l1)
        # print("l1_ndvi: ", l1_ndvi)
        # print("l1_ndvi_mean: ", l1_ndvi_mean)

        return l1+l1_ndvi+l1_ndvi_mean

    def _ndvi(self, data):
        RED = data[:, 30, :, :]
        NIR = data[:, 13, :, :]
        num = (NIR - RED)
        den = (NIR + RED)
        num = torch.clamp(num, 0, 1) + EPSILON
        den = torch.clamp(den, 0, 1) + EPSILON
        # print("num: ", num.max())
        # print("den: ", den.max())

        return num / den

    def _ndvi_mean(self, data):
        RED = torch.mean(data[:, [29, 30, 31], :, :], dim=1, keepdim=True)
        NIR = torch.mean(data[:, [12, 13, 14], :, :], dim=1, keepdim=True)
        num = (NIR - RED)
        den = (NIR + RED)
        num = torch.clamp(num, 0, 1) + EPSILON
        den = torch.clamp(den, 0, 1) + EPSILON
        # print("num: ", num.max())
        # print("den: ", den.max())

        return num / den

class L1Loss_MSEstages(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
        self.alpha = alpha

    def forward(self, output, gt, ):
        l1 = self.l1(output['pred'], gt)

        mse_stages = []
        for i in range(len(output['u_list'])):
            mse_stages.append(self.mse(output['u_list'][i], gt))
        mse_stages = torch.mean(torch.stack(mse_stages))
        loss = l1 + self.alpha * mse_stages
        return loss


class UTeRMLoss(torch.nn.Module):
    def __init__(self, ):
        super(UTeRMLoss, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.w = 0.25
    def forward(self, output, gt):
        l1_fid = self.l1(output['pred'], gt)
        l1_det = self.l1(output['cs_comp'], gt)
        return l1_fid + self.w * l1_det
    def components(self):
        return ['l1_fid', 'l1_det']


# class WINLoss(torch.nn.Module):
#     def __init__(self, ):
#         super(WINLoss, self).__init__()
#         self.alpha = 0.01
#         self.l1 = nn.L1Loss()
#         self.dwt = DWTForward(J=1, wave='haar', mode='zero')  # J=1 level
#
#     def forward(self, output, gt):
#
#         Y = output['pred']
#         G = gt
#
#         loss_pixel = self.l1(Y, G)
#
#         # Wavelet transform (returns approximation and details)
#         Y_LL, Y_H = self.dwt(Y)
#         G_LL, G_H = self.dwt(G)
#
#         # Y_H and G_H are tuples of high-frequency components per level
#         wavelet_loss = sum(self.l1(y, g) for y, g in zip(Y_H[0], G_H[0]))
#
#         return loss_pixel + self.alpha * wavelet_loss

class SSPSRLoss(torch.nn.Module):
    def __init__(self, lamd=1e-3, spatial_tv=True, spectral_tv=True):
        super(SSPSRLoss, self).__init__()
        self.use_spatial_TV = spatial_tv
        self.use_spectral_TV = spectral_tv
        self.fidelity = torch.nn.L1Loss()
        self.spatial = TVLoss(weight=lamd)
        self.spectral = TVLossSpectral(weight=lamd)

    def forward(self, output, gt):
        y = output['pred']
        loss = self.fidelity(y, gt)
        spatial_TV = 0.0
        spectral_TV = 0.0
        if self.use_spatial_TV:
            spatial_TV = self.spatial(y)
        if self.use_spectral_TV:
            spectral_TV = self.spectral(y)
        total_loss = loss + spatial_TV + spectral_TV
        return total_loss

class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    # @staticmethod
    # def gate_loss(gate):
    #     dis = torch.sum(gate,dim=0)
    #     cv = torch.std(dis)/torch.mean(dis)
    #     return cv**2

class SR_SSR_FusionLoss(torch.nn.Module):
    def __init__(self,alpha_ms, alpha_hs,alpha_fusion):
        super(SR_SSR_FusionLoss, self).__init__()
        self.L1 = torch.nn.L1Loss()

    def forward(self, output, gt):
        pred = output['pred']
        l1 = self.L1(pred, gt)
        return l1