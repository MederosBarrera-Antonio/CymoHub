
# TRAINING AND TEST DATA

import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Find training and test files
def find_files(path, vector_idx_test, indicadito_mask="_mask"):
    trainImages = []
    trainMasks = []
    testImages = []
    testMasks = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                num_file = int(file[0:2])
            except:
                num_file = int(file[0:1])
    
            if indicadito_mask not in file:
                if num_file in vector_idx_test:
                    testImages.append(file_path)
                else:
                    trainImages.append(file_path)
            else:
                if num_file in vector_idx_test:
                    testMasks.append(file_path)
                else:
                    trainMasks.append(file_path)
                    
    return trainImages, trainMasks, testImages, testMasks

# Class for data loading
class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, input_width, input_height, num_classes, mean=None, std=None, transforms=None):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        self.input_width = input_width
        self.input_height = input_height
        self.num_classes = num_classes
        self.mean = mean
        self.std = std
        
    def __len__(self):
        if len(self.imagePaths) != len(self.maskPaths):
            print("ERROR: mismatch of image and mask sizes.")
            return None
        else:
            return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]

        if imagePath != maskPath.replace("_mask", "_img"):
            print("ERROR: masks do not match.")
            return None
            
        mat_data_image = scipy.io.loadmat(imagePath)
        filtered_keys_image = [key for key in mat_data_image.keys() if "__" not in key]
        image = mat_data_image[filtered_keys_image[0]]
        
        mat_data_mask = scipy.io.loadmat(maskPath)
        filtered_keys_mask = [key for key in mat_data_mask.keys() if "__" not in key]
        mask = mat_data_mask[filtered_keys_mask[0]]
        
        # DEEP WATER IS EXCHANGED FOR SAND
        mask[mask==5]=4
        
        if (image.shape[0] is not mask.shape[0]) and (image.shape[1] is not mask.shape[1]):
            print("ERROR: images do not match.")
            return None
        
        image = image.transpose((2,0,1))        
        
        mask = np.eye(self.num_classes)[mask]
        mask = mask.transpose((2,0,1))
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        image = image.to(torch.float)
        mask = mask.to(torch.float)
        
        if (image.shape[0] is not self.input_width) or (image.shape[1] is not self.input_height):
            transform = transforms.CenterCrop((self.input_width,self.input_height))
            image = transform(image)
            mask = transform(mask)
        
        if self.mean is not None and self.std is not None:
            transform = transforms.Normalize(mean=self.mean, 
                                             std=self.std)
            image = transform(image)
        
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
            
        return (image, mask)
    

# Mean and std of the dataset
def mean_std_dataset(IMAGE_DATASET_PATH, INPUT_WIDTH, INPUT_HEIGHT, NUM_CLASSES, BATCH_SIZE, PIN_MEMORY):
    
    trainImages, trainMasks, _, _ = find_files(IMAGE_DATASET_PATH, [])

    trainDS = SegmentationDataset(imagePaths = trainImages, 
                                  maskPaths = trainMasks,
                                  input_width = INPUT_WIDTH,
                                  input_height = INPUT_HEIGHT,
                                  num_classes = NUM_CLASSES,
                                  mean=None,
                                  std=None)

    if BATCH_SIZE=="all":
        BATCH_SIZE_TRAIN = len(trainDS)
    else:
        BATCH_SIZE_TRAIN = BATCH_SIZE
    trainLoader = DataLoader(trainDS, 
                             shuffle = False,
                             batch_size = BATCH_SIZE_TRAIN, 
                             pin_memory = PIN_MEMORY)

    n_images = 0
    mean = 0
    for (_, (images, _)) in enumerate(trainLoader):
        n_images += images.size(0)
        mean += images.mean([0, 2, 3]) * images.size(0)
    mean /= n_images

    var = 0
    for images, _ in trainLoader:
        var += ((images - mean.view(1, 8, 1, 1)) ** 2).sum([0, 2, 3])
    var /= (n_images * images.size(2) * images.size(3) - 1)
    std = torch.sqrt(var)

    # mean = tensor([ 0.0111,  0.0390,  0.0318,  0.0031,  0.0016, -0.0103, -0.0103, -0.0154])
    # std  = tensor([0.0070, 0.0132, 0.0138, 0.0077, 0.0051, 0.0036, 0.0033, 0.0042])
    return mean, std


# Training and test data
def data_train_test(IMAGE_DATASET_PATH, INPUT_WIDTH, INPUT_HEIGHT, NUM_CLASSES, BATCH_SIZE, PIN_MEMORY, VECTOR_IDX_TEST):
    mean,std = mean_std_dataset(IMAGE_DATASET_PATH, INPUT_WIDTH, INPUT_HEIGHT, NUM_CLASSES, BATCH_SIZE, PIN_MEMORY)
    
    trainImages, trainMasks, testImages, testMasks = find_files(IMAGE_DATASET_PATH, VECTOR_IDX_TEST)

    trainDS = SegmentationDataset(imagePaths = trainImages, 
                                  maskPaths = trainMasks,
                                  input_width = INPUT_WIDTH,
                                  input_height = INPUT_HEIGHT,
                                  num_classes = NUM_CLASSES,
                                  mean=mean,
                                  std=std)
    testDS = SegmentationDataset(imagePaths = testImages, 
                                 maskPaths = testMasks,
                                 input_width = INPUT_WIDTH,
                                 input_height = INPUT_HEIGHT,
                                 num_classes = NUM_CLASSES,
                                 mean=mean,
                                 std=std)

    if BATCH_SIZE=="all":
        BATCH_SIZE_TRAIN = len(trainDS)
        BATCH_SIZE_TEST = len(testDS)
    else:
        BATCH_SIZE_TRAIN = BATCH_SIZE
        BATCH_SIZE_TEST = BATCH_SIZE

    trainLoader = DataLoader(trainDS, 
                             shuffle = True,
                             batch_size = BATCH_SIZE_TRAIN, 
                             pin_memory = PIN_MEMORY)
    testLoader = DataLoader(testDS, 
                            shuffle = False,
                            batch_size = BATCH_SIZE_TEST,
                            pin_memory = PIN_MEMORY)
    
    return trainLoader, testLoader, trainDS, testDS, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST