import torch

def white_balance_correction(input):
    # Convertir la imagen a formato tensor

    r = input[[0],:,:]
    g = input[[1],:,:]
    b = input[[2],:,:]

    # Calcular el promedio de cada canal de color
    avg_r = torch.mean(r)
    avg_g = torch.mean(g)
    avg_b = torch.mean(b)

    # Calcular el factor de escala para cada canal de color
    scale_r = avg_g / avg_r
    scale_b = avg_g / avg_b

    # Aplicar el balance de blancos corrigiendo los canales
    corrected_r = torch.clamp(r * scale_r, 0, 1)
    corrected_g = g
    corrected_b = torch.clamp(b * scale_b, 0, 1)

    # Combinar los canales corregidos en una imagen
    corrected_image = torch.cat([corrected_r, corrected_g, corrected_b], dim=0)

    return corrected_image


def gamma_correction(input, gamma):
    return torch.pow(input, 1./gamma)


def prisma_correction(input):
    wb_input = white_balance_correction(input)
    gamma_red = 1.4
    gamma_green = 1.7
    gamma_blue = 1.7
    wb_input[0, :, :] = gamma_correction(wb_input[0, :, :], gamma_red)
    wb_input[1, :, :] = gamma_correction(wb_input[1, :, :], gamma_green)
    wb_input[2, :, :] = gamma_correction(wb_input[2, :, :], gamma_blue)
    return wb_input


def scale_min_max(v, new_min=0, new_max=1):
    v_min, v_max = v.min(), v.max()
    return (v - v_min) / (v_max - v_min) * (new_max - new_min) + new_min