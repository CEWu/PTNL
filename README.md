
# Why Is Prompt Tuning for Vision-Language Models Robust to Noisy Labels? [ICCV 2023]



> [**Why Is Prompt Tuning for Vision-Language Models Robust to Noisy Labels?**](https://arxiv.org/abs/2307.11978)<br>
> Cheng-En Wu, Yu Tian, Haichao Yu, Heng Wang, Pedro Morgado, Yu Hen Hu, Linjie Yang

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.11978)


## Introduction

This repo is the official implementation of **Why Is Prompt Tuning for Vision-Language Models Robust to Noisy Labels?**.


## Install

**Setup conda environment (recommended).**:

```bash
############ Conda Environment Installation ############

# Fetch the miniconda script
export HOME=$PWD
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

# Initialize conda
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 yes
conda activate dassl

############ Dassl Installation ############

# Clone the Dassl repository
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/
git reset --hard ac6e44194b2f90e325f477aadd6d9bc3a92ce255

# Establish a new conda environment
conda create -n dassl python=3.7

# Activate the new environment
conda activate dassl

# Install the required dependencies
pip install -r requirements.txt

# Install PyTorch (version 1.11.0 or above) and torchvision
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Set up the Dassl library (No need to rebuild even if the source code changes)
python setup.py develop

############ PTNL Installation ############

# Navigate back to the parent directory
cd ..

# Clone the PTNL repository
git clone https://github.com/CEWu/PTNL
cd PTNL/

# Install necessary packages for CLIP
pip install -r requirements.txt

######## Note ########
# Two symbolic links, `data` and `temp_analyze_results_miltiple`, exist in the repository. It is recommended that these be pointed to locations with sufficient storage capacity.

rm data temp_analyze_results_miltiple # remove the existing links
ln -s ${your_data_path} ./data
ln -s ${your_temp_analyze_results_miltiple_path} ./temp_analyze_results_miltiple

# Installation complete

```


## Datasets

Please follow the instructions at [CoOp Datasets Instructions](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)  to prepare all datasets.


## Training

### Obtain Psuodo-labels and Save Model's logits
```python
CUDA_VISIBLE_DEVICES=0 bash get_info.sh sscaltech101 anay_rn50 end 16 -1 False
```

### Prompt Tuning with Noisy Labels
Training with samples with noisy (2 out of 16 shots training samples are noisy labels.)
```python
CUDA_VISIBLE_DEVICES=0 bash upl_train.sh sscaltech101 rn50_ep50 end 16 16 False True rn50_random_init 2
```

9 arguments listed sequentially as follows:
* dataset config (others in `configs/datasets`)
* model config (only rn50_ep50)
* class token position (end or middle)
* number of context tokens
* number of shots (1, 2, 4, 8, 16)
* class-specific context (False or True)
* CLASS_EQULE True of False
* log tag (only rn50_random_init)
* number of false positive training samples per class

PS. under `scripts`, there are eight scripts (upl_train_*.sh) shring 16 seeds to speed up training process on 8 GPUs.

## Ensemble Testing


### Test with existing files after prompt tuning with noisy labels (2 out of 16 shots training samples are noisy labels.)

```python
bash upl_test_existing_logits.sh sscaltech101 rn50_ep50 end 16 16 False True 2
```

8 arguments listed sequentially as follows:
* dataset config (others in `configs/datasets`)
* model config (only rn50_ep50)
* class token position (end or middle)
* number of context tokens
* number of shots (1, 2, 4, 8, 16)
* class-specific context (False or True)
* CLASS_EQULE True of False
* number of false positive training samples per class

# Citation
If you find our work beneficial for your research, please consider citing:
```
@inproceedings{wu2023ptnl,
    title={Why Is Prompt Tuning for Vision-Language Models Robust to Noisy Labels?},
    author={Cheng-En Wu, Yu Tian, Haichao Yu, Heng Wang, Pedro Morgado, Yu Hen Hu, Linjie Yang},
    booktitle={ICCV},
    year={2023}
}
```

## Acknowlegment

This repository is based on [UPL](https://github.com/tonyhuang2022/UPL).