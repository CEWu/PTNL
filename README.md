
# Prompt Tuning with Noisy Labels for Vision-Language Models


## Introduction

This repo is the official implementation of **Prompt Tuning Noisy with Labels for Vision-Language Models**.


## Install

The code is built on the [CoOp](https://github.com/KaiyangZhou/CoOp) and [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) with commit `ac6e44194b2f90e325f477aadd6d9bc3a92ce255`, so you need to install the dassl environment first. You can follow the [instructions](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install *dassl* as well as *PyTorch*. After that, run `pip install -r requirements.txt` under `PTNL/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP).

**We also prepare all installation commands for you**:

```bash
############ install conda env ############

# Download miniconda script
export HOME=$PWD
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

# Set up conda
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 yes
conda activate dassl

############ install Dassl ############

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/
git reset --hard ac6e44194b2f90e325f477aadd6d9bc3a92ce255

# Create a conda environment
conda create -n dassl python=3.7

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.11.0) and torchvision
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop

############ install PTNL ############

# Enter the directory at the same level as Dassl
cd ..

# Clone this repo
git clone https://github.com/tonyhuang2022/PTNL.git
cd PTNL/

# Install CLIP dependencies
pip install -r requirements.txt

######## attention ########
# We have two soft links, and you can redirect them!
# The `data` is linked to the datasets, and the `temp_analyze_results_miltiple` is linked to the `info`.
# We strongly recommend that you create these two paths on the disk which has enough space, and then use

rm data temp_analyze_results_miltiple # remove the existing file
ln -s ${your_data_path} ./data
ln -s ${your_temp_analyze_results_miltiple_path} ./temp_analyze_results_miltiple

# Finished
```


## Datasets

After that, you can follow the [CoOp Datasets Instructions](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to prepare the datasets.

Then, you can run the code ~

## Training

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

## Acknowlegment

This repository is based on [UPL](https://github.com/tonyhuang2022/UPL).