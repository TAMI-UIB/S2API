import torch.nn
import torch
from torch import nn

from src.model.fusion.GINet_utils import Res_NL, Res_Local, Res_NLv2
from src.model.fusion.DBU_utils import Down, Upsampling

conv_kernel_dict = {"Res_NL": Res_NL ,"Res_Local": Res_Local}


class ClustersUp(nn.Module):
    def __init__(self, ms_channels, hs_channels, classes=5, features=64, **kwargs):
        super(ClustersUp, self).__init__()
        self.classes = classes
        self.ms_channels = ms_channels
        self.hs_channels = hs_channels
        self.mlps = nn.ModuleList([nn.Sequential(*[ nn.Linear(ms_channels, features), nn.ReLU(),  nn.Linear(features, hs_channels), nn.ReLU()]) for _ in range(classes)])

    def forward(self, image, clusters):
        B, C, H, W = image.shape
        hs_image = torch.zeros(B, self.hs_channels, H, W).to(image.device)
        for label in range(self.classes):
            mask = clusters == label
            indices = mask.nonzero(as_tuple=True)
            if indices[0].numel() == 0:
                continue
            pixel_values = image[indices[0], :, indices[2], indices[3]]
            transformed = self.mlps[label](pixel_values)
            hs_image[indices[0], :, indices[2], indices[3]] = transformed
        return hs_image

class NLBPUNet_clusters(nn.Module):
    def __init__(
            self,
            ms_channels,
            hs_channels,
            iter_stages,
            features,
            patch_size,
            window_size,
            kernel_size,
            sampling,
            conv_kernel,
            classes,
            learned=False
    ):
        super(NLBPUNet_clusters, self).__init__()

        self.sampling = sampling
        self.classes=classes
        self.BPKernel = nn.ModuleList([
                conv_kernel_dict[conv_kernel](u_channels=hs_channels, pan_channels=hs_channels,features_channels=features, patch_size=patch_size, window_size=window_size, kernel_size=kernel_size)
                for i in range(iter_stages)
            ])
        # self.upsamp = UpSamp_4_2(hs_channels, ms_channels)
        self.iter_stages = iter_stages
        self.learned=learned
        self.spectral_inter = ClustersUp(ms_channels, hs_channels, classes, features)
        self.clustering_cnn = nn.Sequential(
            nn.Conv2d(ms_channels, 32, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(),
            nn.Conv2d(64, self.classes, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.Softmax(dim=1)  # Probabilidades de pertenencia a cada clúster
        )
        self.downsamp_hs = Down(channels=hs_channels, sampling=sampling)
        self.upsamp_hs = Upsampling(hs_channels, sampling)
        self.downsamp_hs = nn.Sequential(*[Down(channels=hs_channels, sampling=sampling) for _ in range(self.iter_stages)])
        self.upsamp_hs = nn.Sequential(*[Upsampling(hs_channels, sampling) for _ in range(self.iter_stages)])


    def forward(self, pan, hs, ms):
        sampling = ms.size(2) // hs.size(2)
        f = ms
        clusters_probs = self.clustering_cnn(f)
        clusters = torch.argmax(clusters_probs, dim=1, keepdim=True)
        pan_learned = self.spectral_inter(f, clusters)
        u_list = []
        u = nn.functional.interpolate(hs, scale_factor=sampling, mode='bicubic')
        # hs_up = nn.functional.interpolate(hs, scale_factor=sampling, mode='bicubic')

        for i in range(self.iter_stages):
            DBu = self.downsamp_hs[i](u)
            error = self.upsamp_hs[i](DBu - hs)
            u = u + self.BPKernel[i](error, pan_learned)
            u_list.append(u)

        # DBu = self.downsamp_hs[-1](u)
        # if self.learned:
        #     pan_low = nn.functional.interpolate(
        #         nn.functional.interpolate(pan_learned, scale_factor=1. / sampling, mode='bicubic'),
        #         scale_factor=sampling,
        #         mode='bicubic')
        #     Brovey = pan_learned*u-pan_low*hs_up
        # else:
        #     pan_low = nn.functional.interpolate(
        #         nn.functional.interpolate(pan, scale_factor=1. / sampling, mode='bicubic'),
        #         scale_factor=sampling,
        #         mode='bicubic')
        #     Brovey = pan * u - pan_low * hs_up

        return {"pred": u, "u_list": u_list[:-1]} #, "pan": pan_learned,"DBu": DBu-hs,"Brovey": Brovey, "pan_cl":pan, "pan_brovey":pan_low, "hs_up":hs_up}
