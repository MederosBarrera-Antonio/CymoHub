# CymoHub: U-Net, Attention U-Net and Pix2Pix for Seagrass Mapping using WorldView Satellite Imagery

*Note: The results correspond to the article “Comparison of Conventional Machine Learning and Convolutional Deep Learning models for Seagrass Mapping using Satellite Imagery” published in the IEEE JSTARS journal. Please reference the work if used. Thank you.*

![alt text](https://github.com/MederosBarrera-Antonio/CymoHub/blob/main/CymoHub_Image.png "CymoHub Image")

This repository contains data, codes, and final models useful for benthic habitats mapping, especially seagrass, from WorldView-2 multispectral satellite images. Specifically, the code and models of the U-Net, Attention U-Net (U-Net with Attentions Gates in the skip connections) and Pix2Pix (cGAN where the generator is a U-Net and the discriminator is PatchGAN) architectures are published. The PyTorch library with CUDA acceleration has been used for this work (more information in [requirements](#requirements)).

The repository is divided into three folders:  

- Code: PyTorch implementation. Two Jupyter Notebooks are highlighted: one for the U-Net and Attention U-Net models, and another for the Pix2Pix architecture.  
- Data: Data available for the generation of benthic habitat maps. In general folder, you can find a PDF summary of the data, as well as a tutorial on how to obtain it.  
- Models: Final models (.pth) used for the generation of benthic habitat maps.

## Requirements

- Numpy (1.22.4 used)
- Matplotlib (3.5.1 used)
- PyTorch (2.1.2 used with CUDA 12.1 in a NVIDIA GeForce RTX 3050 Ti)
- Pandas (1.4.2 used)
- Seaborn (0.11.2 used)
- Scikit-Learn (1.0.2 used)
	- Base of conda 23.1.0 used
 
## Contact

For any questions or concerns, please contact the author: Antonio Mederos-Barrera (mederosbarrera.antonio@gmail.com)
