#!/bin/bash
for network in resnet-18 resnet-34 resnet-50 vgg-16 vgg-19 densenet-121 inception_v3 mobilenet squeezenet_v1.0 squeezenet_v1.1
do
    python3 gpu_imagenet_bench.py --target=rocm --network=$network
done

    
