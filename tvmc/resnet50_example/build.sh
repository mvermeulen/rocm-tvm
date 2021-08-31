#!/bin/bash
set -x

SAVED_MODELDIR=${SAVED_MODELDIR:="/home/mev/source/rocm-migraphx/saved-models"}

# Compile to TVM runtime
tvmc compile --target "rocm" --output resnet50i1.tar ${SAVED_MODELDIR}/torchvision/resnet50i1.onnx

# Preprocess
python3 preprocess.py

# Running the compiled model
# How do we run a ROCm target?
tvmc run --target --inputs imagenet_cat.npz --output predictions.npz resnet50i1.tar

# Postprocess
python3 postprocess.py
