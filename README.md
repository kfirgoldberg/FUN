# Rethinking FUN: Frequency-Domain Utilization Network  
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/kfir99/FUN/blob/main/notebooks/eFUN_inference_playground.ipynb)

> The search for efficient neural network architectures has gained much focus in recent years, where modern architectures focus not only on accuracy but also on inference time and model size. Here, we present FUN, a family of novel Frequency-domain Utilization Networks. These networks utilize the inherent efficiency of the frequency-domain by working directly in that domain, represented with the Discrete Cosine Transform. Using modern techniques and building blocks such as compound-scaling and inverted-residual layers we generate a set of such networks allowing one to balance between size, latency and accuracy while outperforming competing RGB-based models. Extensive evaluations verifies that our networks present strong alternatives to previous approaches. Moreover, we show that working in frequency domain allows for dynamic compression of the input at inference time without any explicit change to the architecture.

<p float="left">
  <img src="/docs/FUN/acc_vs_size.png" width="300" />
  <img src="/docs/FUN/acc_vs_latency.png" width="300" />
</p>

## Description   
Official PyTorch implementation of "Rethinking FUN: Frequency-domain Utilization Networks".
Based on the [implementation by Ross Wightman](https://github.com/rwightman/pytorch-image-models) 

## Recent Updates
 
## Getting Started

### Installation
- Clone this repo:
``` 
git clone https://github.com/kfir99/FUN.git
cd FUN
```
- Install a conda environment
```
conda create -n torch-env
conda activate torch-env
conda install -c pytorch pytorch torchvision cudatoolkit=10.2 jpeg2dct
conda install pyyaml
```

### Pretrained Models
Pretrained models are available for download

| Path | Accuracy | # Parameters (M) | FPS (V-100, batch size = 1)
| :---: | :----------: | :----------: | :----------: 
|[eFUN](https://drive.google.com/file/d/17GRexna2lMaqELTK-PYgIlwEUk9Hponp/view?usp=sharing)  | 77 | 4.2 | 124 
|[eFUN-L](https://drive.google.com/file/d/1sUs0cDZr9cgMQEsXx2xDDqI0AUMG9oK_/view?usp=sharing)  | 78.8 | 6.2 | 101
|[eFUN-S](https://drive.google.com/file/d/1Vx1rXQ43C3ofWAMhJHtjI8-qDgveEJw3/view?usp=sharing)  | 75.6 | 3.4 | 132
|[eFUN-S+](https://drive.google.com/file/d/1Vx1rXQ43C3ofWAMhJHtjI8-qDgveEJw3/view?usp=sharing) | 73.3 | 2.5 | 145

### Training
#### Preparing your data
Training and validation data should be organized in the following structure: 
* data_dir
    * train
        * class_name_a
            * images
        * class_name_b
            * images
            
            .
        
            .
        
            .
    * validation
        * class_name_a
            * images
        * class_name_b
            * images
        
            .
        
            .
        
            .
#### Command line
**eFUN**
```
./distributed_train.sh \
 4\
 <data_dir>\
 --output <desired output path>\
 --dct\ 
 --model efun\
 --no-prefetcher\
 -b 128\
 --sched step\
 --epochs 450\
 --decay-epochs 2.4\
 --decay-rate .97\
 --opt rmsproptf\
 --opt-eps .001\
 -j 8\
 --warmup-lr 1e-6\
 --weight-decay 1e-5\
 --drop 0.2\ 
 --drop-path 0.2\
 --model-ema\
 --model-ema-decay 0.9999\ 
 --remode pixel\
 --reprob 0.2\
 --lr .048
```

**eFUN-L**
```
./distributed_train.sh\
 4\
 <data_dir>\
 --output <desired output path>\
 --dct\
 --model efun_l\
 --no-prefetcher\
 -b 112\
 --sched step\
 --epochs 450\
 --decay-epochs 2.4\
 --decay-rate .97\
 --opt rmsproptf\
 --opt-eps .001\
 -j 8\
 --warmup-lr 1e-6\
 --weight-decay 1e-5\
 --drop 0.2\
 --drop-path 0.3\
 --model-ema\
 --model-ema-decay 0.9999\
 --remode pixel\
 --reprob 0.2\
 --lr .048
```

**eFUN-S**
```
./distributed_train.sh\
 4\
 <data_dir>\
 --output <desired output path>\
 --dct\
 --model efun_s\
 --no-prefetcher\
 -b 128\
 --sched step\
 --epochs 450\
 --decay-epochs 2.4\
 --decay-rate .97\
 --opt rmsproptf\
 --opt-eps .001\
 -j 8\
 --warmup-lr 1e-6\
 --weight-decay 1e-5\
 --drop 0.2\
 --drop-path 0.2\
 --model-ema\
 --model-ema-decay 0.9999\
 --remode pixel\
 --reprob 0.2\
 --lr .048
```

**eFUN-S+**
```
./distributed_train.sh\
 4\
 <data_dir>\
 --output <desired output path>\
 --dct\
 --model efun_s_plus\
 --no-prefetcher\
 -b 128\
 --sched step\
 --epochs 450\
 --decay-epochs 2.4\
 --decay-rate .97\
 --opt rmsproptf\
 --opt-eps .001\
 -j 8\
 --warmup-lr 1e-6\
 --weight-decay 1e-5\
 --drop 0.2\
 --drop-path 0.2\
 --model-ema\
 --model-ema-decay 0.9999\
 --remode pixel\
 --reprob 0.2\
 --lr .048
```

### Testing
#### Preparing your data
Training and validation data should be organized in the following structure: 
* data_dir
    * class_name_a
        * images
    * class_name_b
        * images
        
        .
    
        .
    
        .
#### Command line
```
python validate.py\
 <path to test data>\
 --model <efun/efun_l/efun_s/efun_s_plus>\
 --checkpoint <path to trained model>\
 --dct\
 --no-prefetcher\
 -j 32\
 -b 256
```

## Repository structure
Files added specifically for eFUN are marked in **bold** 

| Path | Description <img width=200>
| :--- | :---
| FUN | Repository root folder
| &boxvr;&nbsp; convert | Folder containing code for conversion from MXNET weights
| &boxvr;&nbsp; docs | Folder containing documentation of the results and graphs use in the repository
| &boxvr;&nbsp; notebooks | Folder with runnable notebooks for experimenting with the dataset
| &boxv;&nbsp; &boxur;&nbsp; **eFUN_inference_playground.ipynb** | A basic notebook to run inference on some dataset using a pretrained eFUN model
| &boxvr;&nbsp; results | Folder with reported results for original TIMM repository models 
| &boxv;&nbsp; tests | Folder with some tests for original TIMM repository models
| &boxv;&nbsp; timm | Folder with actual TIMM pip package modules
| &boxv;&nbsp; &boxvr;&nbsp; data | Folder with implementation of all data-related functions
| &boxv;&nbsp; &boxur;&nbsp; **rgb2dct.py** | implementation of conversion from RGB image to its DCT representation
| &boxv;&nbsp; &boxvr;&nbsp; loss | Folder with implementations of different loss functions
| &boxv;&nbsp; &boxvr;&nbsp; models | Folder with implementation of all models provided in this repository
| &boxv;&nbsp; &boxur;&nbsp; **eFUN.py** | Implementation of all eFUN models
| &boxv;&nbsp; &boxvr;&nbsp; optim | Folder with implementations of different optimizers
| &boxv;&nbsp; &boxvr;&nbsp; scheduler | Folder with implementations of different training schedulers
| &boxv;&nbsp; &boxvr;&nbsp; utils | Folder with implementations of utility functions


## TODOs
- [ ] sota-bench integration 

## Citation

