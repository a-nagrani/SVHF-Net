# SVHF-Net
# SVHF-Net for Cross-modal binary matching

This directory contains code to import and evaluate the static SVHF-Net model trained on the VoxCeleb and VGGFace datasets as described in the [paper](http://www.robots.ox.ac.uk/~vgg/publications/2018/Nagrani18a/nagrani18a.pdf): 

``` 
A. Nagrani, S. Albanie, A. Zisserman, Seeing Voices and Hearing Faces: Cross-modal biometric matching, 
CVPR, 2018
``` 
Further details can be found [here](file:///Users/arshanagrani/Dropbox/Websites/project_pages/seeing-voices-cvpr/index.html).

### Prerequisites

To use the models first install the MatConvNet framework.  Instructions can 
be found [here](http://www.vlfeat.org/matconvnet/).


### Installing

To install, follow these steps: 

1. Install and compile matconvnet by following instructions [here](http://www.vlfeat.org/matconvnet/install/). 

2. Setup paths:

```
setup_SVHFNet

```
3. You can then run the demo script provided to import and test the model.

```
test_SVHFNet

```
 
## Dataset 
This model has been trained on static face images from the VoxCeleb and VGGFace datasets, and audio segments from the VoxCeleb dataset.
The VoxCeleb dataset can be downloaded directly from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/). Cropped face images can be downloaded from [here](file:///Users/arshanagrani/Dropbox/Websites/project_pages/seeing-voices-cvpr/index.html).

## Citation
If you use this code then please cite:

```bibtex
  @InProceedings{Nagrani18a,
                    author       = "Nagrani, A. and Albanie, S. and Zisserman, A.",
                    title        = "Seeing Voices and Hearing Faces: Cross-modal biometric matching",
                    booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
                    year         = "2018",
                  }
```
