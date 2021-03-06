# img2mml_resnet

This repository consists of scripts to run `Image2MML` model. 

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
Image preprocessing:
Instead of using images as im=nput, we will first convert them to tensors to make the training faster.
```
python3 preprocessing/preprocess_images.py
```

## To run the model: 

We will be using `PyTorch data_parallelism` to make training faster by exploiting multiple GPUs. 

```
python3 main_ddp.py --local_rank 0 --batch_size 128 --epochs 100
```
