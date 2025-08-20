
# CLASS WEIGHTS FROM DATA

import torch

def obtain_class_weights(trainLoader, NUM_CLASSES, DEVICE):

    class_counts = torch.zeros(NUM_CLASSES)

    for (_, (_, y)) in enumerate(trainLoader):
        for _, targets in enumerate(y):
            for i in range(targets.shape[0]):
                class_counts[i] += targets[i,:,:].sum()


    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()
    class_weights = class_weights.to(DEVICE)
    
    return class_weights