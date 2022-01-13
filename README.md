# VRDLHW4 Super resolution
## enlarge images to 3x

## Dataset:
 - 192 training images
 - 14 testing images

## Dataset Preparation
I split 291 images to 0.8: 0.2  
232 for training set and 59 for validation set.

## Requirements
```
conda create --name hw4 python=3.7
conda activate hw4
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install Pillow
conda install opencv-python
conda install tensorboard
```

## Train 
```python train_srresnet.py```

## Inference
```python inference.py```

## My training weight
Download the whole tar file checkpoint_srresnet.pth.tar  
https://drive.google.com/file/d/1i3eWpareo1f1Wxler7Sy9S-lkZZb0XsA/view?usp=sharing  
You can easily use the weight by modifying the path in ```inference.py```

## Reference:  
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
https://github.com/twtygqyy/pytorch-SRResNet
