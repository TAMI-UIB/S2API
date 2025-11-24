import numpy as np
import torch
from torch.nn.functional import fold, unfold


class CreatePatches():
    def __init__(self, data, patch_size, overlapping=None):
        self.patch_size = patch_size
        self.overlapping = overlapping
        self.N = data.size(0)
        self.C = data.size(1)
        self.H = data.size(2)
        self.W = data.size(3)

    def do_patches(self, data):
        if self.overlapping is None:
            data_shape = data.size
            if not data_shape(2) % self.patch_size == 0 and not data_shape(3) % self.patch_size == 0:
                raise ValueError('Data shape must be divisible by patch size')
            patches = unfold(data, kernel_size=self.patch_size, stride=self.patch_size)
            patch_num = patches.size(2)
            patches = patches.permute(0, 2, 1).view(data_shape(0), -1, data_shape(1), self.patch_size, self.patch_size)
            return torch.reshape(patches, (data_shape(0) * patch_num, data_shape(1), self.patch_size, self.patch_size))
        else:
            patches = []
            for i in range(0, data.shape[2] - self.patch_size + 1, self.patch_size - self.overlapping):
                for j in range(0, data.shape[3] - self.patch_size + 1, self.patch_size - self.overlapping):
                    patch = data[:, :, i:i + self.patch_size, j:j + self.patch_size]
                    patches.append(patch)
            batched_patches = torch.cat(patches, axis=0)
            return batched_patches

    def undo_patches(self, data):
        if self.overlapping is None:
            patches = data.reshape(self.N, data.size(0), data.size(1), self.patch_size, self.patch_size)
            patches = patches.view(self.N, data.size(0), data.size(1) * self.patch_size * self.patch_size).permute(0, 2, 1)
            return fold(patches, (self.W, self.H), kernel_size=self.patch_size, stride=self.patch_size)
        else:
            N = data.size(0)
            patch_size = self.patch_size
            overlap = self.overlapping
            reconstructed_tensor = torch.zeros((1, self.C, self.H, self.W)).to(data.device)
            for patch_idx, patch in enumerate(data):
                i_start = (patch_idx // ((self.W - overlap) // (patch_size - overlap))) * (patch_size - overlap)
                j_start = (patch_idx % ((self.W - overlap) // (patch_size - overlap))) * (patch_size - overlap)
                i_end = i_start + patch_size
                j_end = j_start + patch_size
                imin = i_start + overlap // 2
                imax = i_end - overlap // 2
                iimin = overlap // 2
                iimax = self.patch_size - overlap // 2
                jmin = j_start + overlap // 2
                jmax = j_end - overlap // 2
                jjmin = overlap // 2
                jjmax = self.patch_size - overlap // 2
                if patch_idx < int(np.sqrt(N)):
                    imin = i_start
                    iimin = 0
                if patch_idx % int(np.sqrt(N)) == 0:
                    jmin = j_start
                    jjmin = 0
                if patch_idx % int(np.sqrt(N)) == int(np.sqrt(N)) - 1:
                    jmax = j_end
                    jjmax = self.patch_size
                if patch_idx >= int(np.sqrt(N)) * (int(np.sqrt(N)) - 1):
                    imax = i_end
                    iimax = self.patch_size
                reconstructed_tensor[0, :, imin:imax, jmin:jmax] += patch[:, iimin:iimax, jjmin:jjmax]
            return reconstructed_tensor
