#!/bin/bash
#
# Program example using TVMC.
#
# Adapted from: https://tvm.apache.org/docs/tutorials/get_started/tvmc_command_line_driver.html
#
# Note: This seems to work for the "llvm" CPU target but not when changing target to "rocm"
set -x

SAVED_MODELDIR=${SAVED_MODELDIR:="/home/mev/source/rocm-migraphx/saved-models"}

# Compile to TVM runtime
tvmc -v compile --target "llvm" --output resnet50i1.tar ${SAVED_MODELDIR}/torchvision/resnet50i1.onnx

# Preprocess
python3 preprocess.py

# Running the compiled model
# What is needed to run a rocm target?
tvmc -v run --inputs imagenet_cat.npz --output predictions.npz resnet50i1.tar

# Postprocess
python3 postprocess.py
