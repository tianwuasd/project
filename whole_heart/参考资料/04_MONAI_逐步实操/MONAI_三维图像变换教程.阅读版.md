# MONAI_三维图像变换教程

> 从官方 Notebook 导出的阅读版，未运行代码，输出与图像未展开。


Copyright (c) MONAI Consortium  
Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at  
&nbsp;&nbsp;&nbsp;&nbsp;http://www.apache.org/licenses/LICENSE-2.0  
Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.

# Overview

This notebook introduces you MONAI's transformation module for 3D images.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/modules/3d_image_transforms.ipynb)

## Setup environment


```python
!python -c "import monai" || pip install -q "monai-weekly[nibabel]"
!python -c "import matplotlib" || pip install -q matplotlib
```


## Setup imports


```python
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
)
from monai.config import print_config
from monai.apps import download_and_extract
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob

print_config()
```


## Setup data directory

You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.  
This allows you to save results and reuse downloads.  
If not specified a temporary directory will be used.


```python
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(f"root dir is: {root_dir}")
```


## Download dataset

Downloads and extracts the dataset.  
The dataset comes from http://medicaldecathlon.com/.


```python
resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
data_dir = os.path.join(root_dir, "Task09_Spleen")
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir, md5)
```


## Set MSD Spleen dataset path

The following groups images and labels from `Task09_Spleen/imagesTr` and `Task09_Spleen/labelsTr` into pairs.


```python
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_data_dicts, val_data_dicts = data_dicts[:-9], data_dicts[-9:]
```


The image file names are organised into a list of dictionaries.


```python
train_data_dicts[0]
```


The list of data dictionaries, `train_data_dicts`,
could be used by PyTorch's data loader.

For example,

```python
from torch.utils.data import DataLoader

data_loader = DataLoader(train_data_dicts)
for training_sample in data_loader:
    # run the deep learning training with training_sample
```

The rest of this tutorial presents a set of "transforms"
converting `train_data_dict` into data arrays that will
eventually be consumed by the deep learning models.

## Load the NIfTI files

One design choice of MONAI is that it provides not only the high-level workflow components,
but also relatively lower level APIs in their minimal functioning form.

For example, a `LoadImage` class is a simple callable wrapper of the underlying `Nibabel` image loader.
After constructing the loader with a few necessary system parameters,
calling the loader instance with a `NIfTI` filename will return the image data arrays,
as well as the metadata -- such as affine information and voxel sizes.


```python
loader = LoadImage(dtype=np.float32, image_only=True)
```



```python
image = loader(train_data_dicts[0]["image"])
# print(f"input: {train_data_dicts[0]['image']}")
print(f"image shape: {image.shape}")
print(f"image affine:\n{image.meta['affine']}")
print(f"image pixdim:\n{image.pixdim}")
```


Oftentimes, we want to load a group of inputs as a training sample.
For example training a supervised image segmentation network requires a pair of image and label as a training sample.

To ensure a group of inputs are being preprocessed consistently,
MONAI also provides dictionary-based interfaces for the minimal functioning transforms.

`LoadImaged` is the corresponding dict-based version of `LoadImage`:


```python
loader = LoadImaged(keys=("image", "label"), image_only=False)
```



```python
data_dict = loader(train_data_dicts[0])
# print(f"input:, {train_data_dicts[0]}")
print(f"image shape: {data_dict['image'].shape}")
print(f"label shape: {data_dict['label'].shape}")
print(f"image pixdim:\n{data_dict['image'].pixdim}")
```



```python
image, label = data_dict["image"], data_dict["label"]
plt.figure("visualize", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 30], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 30])
plt.show()
```


## Ensure the first dimension is channel

Most of MONAI's image transformations assume that the input data has the shape:  
`[num_channels, spatial_dim_1, spatial_dim_2, ... ,spatial_dim_n]`  
so that they could be interpreted consistently (as "channel-first" is commonly used in PyTorch).  
Here the input image has shape `(512, 512, 55)` which isn't in the acceptable shape (missing the channel dimension),  
we therefore create a transform which is called to update the shape:


```python
ensure_channel_first = EnsureChannelFirstd(keys=["image", "label"])
datac_dict = ensure_channel_first(data_dict)
print(f"image shape: {datac_dict['image'].shape}")
```


Now we are ready to do some intensity and spatial transforms.

## Reorientation to a designated axes codes

Sometimes it is nice to have all the input volumes in a consistent axes orientation.  
The default axis labels are Left (L), Right (R), Posterior (P), Anterior (A), Inferior (I), Superior (S).  
The following transform is created to reorientate the volumes to have 'Posterior, Left, Inferior' (PLI) orientation (To ensure the spatial axes are processed consistently across the subjects, this orientation transform should be put before any anisotropic spatial transforms):


```python
orientation = Orientationd(keys=["image", "label"], axcodes="PLI")
```



```python
data_dict = orientation(datac_dict)
print(f"image shape: {data_dict['image'].shape}")
print(f"label shape: {data_dict['label'].shape}")
print(f"image affine after Spacing:\n{data_dict['image'].meta['affine']}")
print(f"label affine after Spacing:\n{data_dict['label'].meta['affine']}")
```



```python
image, label = data_dict["image"], data_dict["label"]
plt.figure("visualise", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[0, :, :, 30], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0, :, :, 30])
plt.show()
```


## Resample to a consistent voxel size

The input volumes might have different voxel sizes.  
The following transform is created to normalise the volumes to have (1.5, 1.5, 5.) millimetre voxel size.  
The transform is set to read the original voxel size information from `data_dict['image'].affine`,  
which is from the corresponding NIfTI file, loaded earlier by `LoadImaged`.


```python
spacing = Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 5.0), mode=("bilinear", "nearest"))
```



```python
data_dict = spacing(data_dict)
print(f"image shape: {data_dict['image'].shape}")
print(f"label shape: {data_dict['label'].shape}")
print(f"image affine after Spacing:\n{data_dict['image'].meta['affine']}")
print(f"label affine after Spacing:\n{data_dict['label'].meta['affine']}")
```


To track the spacing changes, the data_dict was updated by `Spacingd`:
* An `image.meta['original_affine']` key is added to the `data_dict`, logs the original affine.
* An `image.affine` key is updated to have the current affine.


```python
image, label = data_dict["image"], data_dict["label"]
plt.figure("visualise", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[0, :, :, 30], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0, :, :, 30])
plt.show()
```


## Random affine transformation

The following affine transformation is defined to output a (300, 300, 50) image patch.  
The patch location is randomly chosen in a range of (-40, 40), (-40, 40), (-2, 2) in x, y, and z axes respectively.  
The translation is relative to the image centre.  
The 3D rotation angle is randomly chosen from (-45, 45) degrees around the z axis, and 5 degrees around x and y axes.  
The random scaling factor is randomly chosen from (1.0 - 0.15, 1.0 + 0.15) along each axis.


```python
rand_affine = RandAffined(
    keys=["image", "label"],
    mode=("bilinear", "nearest"),
    prob=1.0,
    spatial_size=(300, 300, 50),
    translate_range=(40, 40, 2),
    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
    scale_range=(0.15, 0.15, 0.15),
    padding_mode="border",
)
rand_affine.set_random_state(seed=123)
```


You can rerun this cell to generate a different randomised version of the original image.


```python
affined_data_dict = rand_affine(data_dict)
print(f"image shape: {affined_data_dict['image'].shape}")

image, label = affined_data_dict["image"][0], affined_data_dict["label"][0]
plt.figure("visualise", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 23], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 23])
plt.show()
```


## Random elastic deformation

Similarly, the following elastic deformation is defined to output a (300, 300, 10) image patch.  
The image is resampled from a combination of affine transformations and elastic deformations.  
`sigma_range` controls the smoothness of the deformation (larger than 15 could be slow on CPU)  
`magnitude_range` controls the amplitude of the deformation (large than 500, the image becomes unrealistic).


```python
rand_elastic = Rand3DElasticd(
    keys=["image", "label"],
    mode=("bilinear", "nearest"),
    prob=1.0,
    sigma_range=(5, 8),
    magnitude_range=(100, 200),
    spatial_size=(300, 300, 10),
    translate_range=(50, 50, 2),
    rotate_range=(np.pi / 36, np.pi / 36, np.pi),
    scale_range=(0.15, 0.15, 0.15),
    padding_mode="border",
)
rand_elastic.set_random_state(seed=123)
```


You can rerun this cell to generate a different randomised version of the original image.


```python
deformed_data_dict = rand_elastic(data_dict)
print(f"image shape: {deformed_data_dict['image'].shape}")

image, label = deformed_data_dict["image"][0], deformed_data_dict["label"][0]
plt.figure("visualise", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 5], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 5])
plt.show()
```


## Cleanup data directory

Remove directory if a temporary was used.


```python
if directory is None:
    shutil.rmtree(root_dir)
```
