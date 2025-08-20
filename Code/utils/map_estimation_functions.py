
# MAP ESTIMATION FUNCTIONS

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
from scipy.stats import mode

def numpy_to_torch(image, mean, std):
    image = image.transpose((2,0,1))        
    image = torch.from_numpy(image)
    image = image.to(torch.float)
    transform = transforms.Normalize(mean=mean, 
                                     std=std)
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def predict_model(image, model, DEVICE):
    image_  = image.to(DEVICE)
    pred = model(image_)
    pred = torch.argmax(pred[0,:,:,:], dim=0)
    pred.cpu().detach().numpy()
    return pred

def sliding_window(matrix, window_size, stride, model, DEVICE):
    shape = matrix.shape
    rows = (shape[2] - window_size[0]) // stride + 1
    cols = (shape[3] - window_size[1]) // stride + 1
    
    all_values = np.zeros((int(np.ceil((window_size[0]**2)/stride)),
                          shape[2],
                          shape[3]),
                          dtype=float)
    
    indx_func = 0
    for i in tqdm(range(0, rows * stride, stride)):
        for j in range(0, cols * stride, stride):
            all_values[indx_func, i:i+window_size[0], j:j+window_size[1]] = predict_model(matrix[:, :, i:i+window_size[0], j:j+window_size[1]], model, DEVICE).cpu().detach().numpy()
            indx_func += 1
            if indx_func == (int(np.ceil((window_size[0]**2)/stride))-1):
                indx_func = 0
    return all_values

def mode_no_zeros(matrix):
    flattened = matrix.flatten()
    non_zero_values = flattened[flattened != 0]

    if len(non_zero_values) == 0:
        return 0
    else:
        mode_result = mode(non_zero_values)
        if isinstance(mode_result.mode, np.ndarray):
            return mode_result.mode[0]
        else:
            return mode_result.mode

def obtain_mode_no_zeros(matrix_3d):
    shape = matrix_3d.shape
    result = np.zeros((shape[1], shape[2]), dtype=matrix_3d.dtype)

    for i in tqdm(range(shape[1])):
        for j in range(shape[2]):
            result[i, j] = mode_no_zeros(matrix_3d[:, i, j])

    return result