# img2mml_resnet

This repository consists of scripts to run `image2mml` model. 

## Requirements 
python >=3.7

PyTorch == 1.9.0

torchvision == 0.10.0

torchaudio == 0.9.0

cudatoolkit >= 11.1

torchtext

## Preprocessing
Before moving forward, first download the dataset from `<data_source>`. Save the images under folder `data/images` and MathML as `data/original_mml.txt` under `data` folder.

MathML preprocessing:
This script will remove the unnecessary tokens from the MathML equation, required only to represent it beautifully.
```
python3 preprocessing/preprocess_mml.py
```


```
python3 preprocessing/preprocess_images.py
```

```
python3 main_ddp.py --gpu_num 1
```
