import math

import torch


class IndexCalculator(object):
    def __init__(self, list_cw, list_fwhm, aggregation_mode='central', verbose=False):
        self.list_cw = list_cw
        self.list_fwhm = list_fwhm

        self.red = 740
        self.green = 490
        self.nir = 842
        self.nir1 = 838.5
        self.nir2 = 838.5
        self.swir1 = 1612
        self.swir2 = 2190

        self.aggregation_mode = aggregation_mode
        self.aggregation_methods = {'central': self.central,
                                    'mean': self.mean,
                                    'weighted_mean': self.weighted_mean,
                                    'gaussian_mean': self.gaussian_mean,
                                    }

        self.verbose = verbose

    def get_red(self, data, mode=None):
        mode = mode if mode else self.aggregation_mode
        red_bands = self.get_bands(self.red)
        if self.verbose:
            print("red_bands: ", red_bands)
        return self.aggregation(data, red_bands, mode=mode)

    def get_green(self, data, mode=None):
        mode = mode if mode else self.aggregation_mode
        green_bands = self.get_bands(self.green)
        if self.verbose:
            print("green_bands: ", green_bands)
        return self.aggregation(data, green_bands, mode=mode)

    def get_nir(self, data, mode=None):
        mode = mode if mode else self.aggregation_mode
        nir_bands = self.get_bands(self.nir)
        if self.verbose:
            print("nir_bands: ", nir_bands)
        return self.aggregation(data, nir_bands, mode=mode)

    def get_nir1(self, data, mode=None):
        mode = mode if mode else self.aggregation_mode
        nir1_bands = self.get_bands(self.nir1)
        if self.verbose:
            print("nir1_bands: ", nir1_bands)
        return self.aggregation(data, nir1_bands, mode=mode)

    def get_nir2(self, data, mode=None):
        mode = mode if mode else self.aggregation_mode
        nir2_bands = self.get_bands(self.nir2)
        if self.verbose:
            print("nir2_bands: ", nir2_bands)
        return self.aggregation(data, nir2_bands, mode=mode)

    def get_swir1(self, data, mode=None):
        mode = mode if mode else self.aggregation_mode
        swir1_bands = self.get_bands(self.swir1)
        if self.verbose:
            print("swir1_bands: ", swir1_bands)
        return self.aggregation(data, swir1_bands, mode=mode)

    def get_swir2(self, data, mode=None):
        mode = mode if mode else self.aggregation_mode
        swir2_bands = self.get_bands(self.swir2)
        if self.verbose:
            print("swir2_bands: ", swir2_bands)
        return self.aggregation(data, swir2_bands, mode=mode)

    def get_bands(self, nm) -> list:
        bands = []
        for i, (c, w) in enumerate(zip(self.list_cw, self.list_fwhm)):
            if type(nm) is int or type(nm) is float:
                if c - w <= nm <= c + w:
                    bands.append(i)
            elif type(nm) is list:
                for b_nm in nm:
                    if c - w <= b_nm <= c + w:
                        bands.append(i)
        bands = list(set(bands))
        bands.sort()
        return bands

    def get_central_wavelength(self, nm):
        bands = self.get_bands(nm)
        central_band = len(bands)//2
        print(bands[central_band], str(self.list_cw))
        return self.list_cw[bands[central_band]]

    def aggregation(self, data, nm_bands, mode):
        if nm_bands:
            return self.aggregation_methods[mode](data, nm_bands)
        raise Exception

    def central(self, data, nm_bands):
        central_band = len(nm_bands)//2
        return data[:, nm_bands[central_band], :, :].unsqueeze(0)

    def mean(self, data, nm_bands):
        return torch.mean(data[:, nm_bands, :, :], dim=1, keepdim=True)

    def weighted_mean(self, data, nm_bands):
        weights = [math.exp(-self.list_fwhm[band]) for band in nm_bands]
        weights = [w / sum(weights) for w in weights]
        return torch.sum(torch.einsum('i,bixy->bixy', weights, data), dim=1, keepdim=True)

    def gaussian_mean(self, data, nm_bands):
        start = self.list_cw[nm_bands[1]] - self.list_fwhm
        end = self.list_cw[nm_bands[0]] + self.list_fwhm
        mu = torch.tensor((start + end) / 2).view(1, -1, 1, 1)
        sigma = torch.tensor((end - start) / 2).view(1, -1, 1, 1)
        g = torch.exp(-0.5 * ((torch.tensor(data) - mu) / sigma) ** 2)
        weights = g / g.sum()
        return torch.sum(torch.einsum('bixy,bixy->bixy', weights, data), dim=1, keepdim=True)

    def NDVI(self, data, mode=None):
        RED = self.get_red(data, mode=mode)
        NIR = self.get_nir(data, mode=mode)

        return (NIR - RED) / (NIR + RED)

    def NDMI(self, data, mode=None):
        NIR = self.get_nir(data, mode=mode)
        SWIR_1 = self.get_swir1(data, mode=mode)

        return (NIR - SWIR_1) / (NIR + SWIR_1)

    def NDWI(self, data, mode=None):
        GREEN = self.get_green(data, mode=mode)
        NIR = self.get_nir(data, mode=mode)

        return (GREEN - NIR) / (GREEN + NIR)

    def WRI(self, data, mode=None):
        GREEN = self.get_green(data, mode=mode)
        RED = self.get_red(data, mode=mode)
        NIR = self.get_nir(data, mode=mode)
        SWIR_2 = self.get_swir2(data, mode=mode)

        return (GREEN + RED) / (NIR + SWIR_2)

    def MNDWI(self, data, mode=None):
        GREEN = self.get_green(data, mode=mode)
        SWIR_2 = self.get_swir2(data, mode=mode)

        return (GREEN - SWIR_2) / (GREEN + SWIR_2)

    def SR(self, data, mode=None):
        NIR = self.get_nir(data, mode=mode)
        RED = self.get_red(data, mode=mode)

        return NIR / RED

    def RNDVI(self, data, mode=None):
        NIR = self.get_nir(data, mode=mode)
        RED = self.get_red(data, mode=mode)
        NDVI = self.NDVI(data, mode=mode)

        return (RED - NDVI) / (RED + NIR)

    def AWEI(self, data, mode=None):
        GREEN = self.get_green(data, mode=mode)
        SWIR_1 = self.get_swir1(data, mode=mode)
        NIR = self.get_nir(data, mode=mode)
        SWIR_2 = self.get_swir2(data, mode=mode)

        return 4 * (GREEN - SWIR_2) - (0.25 * NIR + 2.75 * SWIR_1)

    def INDVI(self, data, mode=None):
        RED = self.get_red(data, mode=mode) * 255
        SWIR_1 = self.get_swir1(data, mode=mode) * 255
        SWIR_2 = self.get_swir2(data, mode=mode) * 255

        indvi = (0.9 * RED + SWIR_1 - 1.9 * SWIR_2 - 155)/(1.1 * RED - SWIR_1 + 1.9 * SWIR_2 + 155)
        return indvi / 255

    def MINDVI(self, data, mode=None):
        NIR = self.get_nir(data, mode=mode) * 255
        SWIR_1 = self.get_swir1(data, mode=mode) * 255
        SWIR_2 = self.get_swir2(data, mode=mode) * 255

        mindvi = (0.9 * NIR + SWIR_1 - 1.9 * SWIR_2 - 155)/(1.1 * NIR - SWIR_1 + 1.9 * SWIR_2 + 155)
        return mindvi / 255

    def PI(self, data, mode=None):
        RED = self.get_red(data, mode=mode)
        NIR = self.get_nir(data, mode=mode)

        return NIR / (RED + NIR)

    def FDI(self, data, mode=None):
        NIR = self.get_nir(data, mode=mode)
        RED = self.get_red(data, mode=mode)
        SWIR1 = self.get_swir1(data, mode=mode)

        l_nir = self.get_central_wavelength(self.nir)
        l_red = self.get_central_wavelength(self.red)
        l_swir1 = self.get_central_wavelength(self.swir1)
        print(f"({l_nir} - {l_red}) / ({l_swir1} - {l_red})")
        lambda_component = (l_nir - l_red) / (l_swir1 - l_red)

        return NIR - (RED + (SWIR1 - RED) * lambda_component * 10)

    # def HI(self, data, mode=None):

    def FAI(self, data, mode=None):
        NIR = self.get_nir(data, mode=mode)
        RED = self.get_red(data, mode=mode)
        SWIR1 = self.get_swir1(data, mode=mode)

        l_nir = self.get_central_wavelength(self.nir)
        l_red = self.get_central_wavelength(self.red)
        l_swir1 = self.get_central_wavelength(self.swir1)
        lambda_component = (l_nir - l_red) / (l_swir1 - l_red)

        return NIR - RED - (SWIR1 - RED)*lambda_component

    def KNDVI(self, data, mode=None):
        NDVI = self.NDVI(data, mode=mode)

        return torch.pow(torch.tanh(NDVI), 2)

    def MARI(self, data, mode=None):
        GREEN = self.get_green(data, mode=mode)
        NIR1 = self.get_nir1(data, mode=mode)
        NIR2 = self.get_nir2(data, mode=mode)

        return 1/GREEN - NIR2/NIR1

    def INTER3(self, data, mode=None):
        index1 = self.index1(data, mode=mode)
        index2 = self.index2(data, mode=mode)
        index3 = self.index3(data, mode=mode)

        return index1 * index2 * index3

    def index1(self, data, mode=None):
        Ri = self.aggregation(data, self.get_bands(781), mode=mode)
        Rj = self.aggregation(data, self.get_bands(951), mode=mode)

        res = torch.pow(Ri, 2) - Rj
        mean = torch.mean(res)
        std = torch.std(res)

        return torch.where(res > mean + 2.2 * std, 0, 1)

    def index2(self, data, mode=None):
        Ri = self.aggregation(data, self.get_bands(596), mode=mode)
        Rj = self.aggregation(data, self.get_bands(719), mode=mode)

        res = torch.pow(Ri, 2) - torch.pow(Rj, 2)
        mean = torch.mean(res)
        std = torch.std(res)

        return torch.where(res > mean + 2.2 * std, 0, 1)

    def index3(self, data, mode=None):
        Ri = self.aggregation(data, self.get_bands(492), mode=mode)
        Rj = self.aggregation(data, self.get_bands(719), mode=mode)

        res = Ri - Rj
        mean = torch.mean(res)
        std = torch.std(res)

        return torch.where(res > mean - 0.6 * std, 0, 1)
