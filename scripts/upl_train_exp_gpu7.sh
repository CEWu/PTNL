#!/bin/bash

CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh sscaltech101 rn50_ep50 end 16 16 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh sscaltech101 rn50_ep50 end 16 16 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh sscaltech101 rn50_ep50 end 16 16 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh sscaltech101 rn50_ep50 end 16 16 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssoxford_pets rn50_ep50 end 16 16 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssoxford_pets rn50_ep50 end 16 16 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssoxford_pets rn50_ep50 end 16 16 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssoxford_pets rn50_ep50 end 16 16 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssucf101 rn50_ep50 end 16 16 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssucf101 rn50_ep50 end 16 16 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssucf101 rn50_ep50 end 16 16 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssucf101 rn50_ep50 end 16 16 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssoxford_flowers rn50_ep50 end 16 16 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssoxford_flowers rn50_ep50 end 16 16 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssoxford_flowers rn50_ep50 end 16 16 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssoxford_flowers rn50_ep50 end 16 16 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssstanford_cars rn50_ep50 end 16 16 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssstanford_cars rn50_ep50 end 16 16 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssstanford_cars rn50_ep50 end 16 16 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=7 bash upl_train_16.sh ssstanford_cars rn50_ep50 end 16 16 False True rn50_random_init 8