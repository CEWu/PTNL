#!/bin/bash

CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh sscaltech101 rn50_ep50 end 16 16 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh sscaltech101 rn50_ep50 end 16 16 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh sscaltech101 rn50_ep50 end 16 16 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh sscaltech101 rn50_ep50 end 16 16 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssoxford_pets rn50_ep50 end 16 16 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssoxford_pets rn50_ep50 end 16 16 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssoxford_pets rn50_ep50 end 16 16 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssoxford_pets rn50_ep50 end 16 16 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssucf101 rn50_ep50 end 16 16 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssucf101 rn50_ep50 end 16 16 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssucf101 rn50_ep50 end 16 16 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssucf101 rn50_ep50 end 16 16 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssoxford_flowers rn50_ep50 end 16 16 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssoxford_flowers rn50_ep50 end 16 16 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssoxford_flowers rn50_ep50 end 16 16 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssoxford_flowers rn50_ep50 end 16 16 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssstanford_cars rn50_ep50 end 16 16 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssstanford_cars rn50_ep50 end 16 16 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssstanford_cars rn50_ep50 end 16 16 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=2 bash upl_train_6.sh ssstanford_cars rn50_ep50 end 16 16 False True rn50_random_init 8