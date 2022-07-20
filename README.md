# img2mml_resnet

This repository consists of the scripts to run `image2mml` model. 

## Requirements 
python >=3.7

PyTorch == 1.9.0

torchvision == 0.10.0

torchaudio == 0.9.0

cudatoolkit >= 11.1

torchtext


```
python3 preprocessing/preprocess_mml.py
```

```
python3 preprocessing/preprocess_images.py
```

```
python3 main_ddp.py --gpu_num 1
```
