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
python3 postprocess.py

# Tune the model
tvmc tune --target "llvm" --output resnet50i1-autotuner.json ${SAVED_MODELDIR}/torchvision/resnet50i1.onnx

# Compile with tuned data
tvmc compile --target "llvm" --tuning-records resnet50i1-autotuner.json --output resnet50i1-tuned.tar ${SAVED_MODELDIR}/torchvision/resnet50i1.onnx

# Run the compiled model
tvmc -v run --inputs imagenet_cat.npz --output predictions.npz resnet50i1-tuned.tar
python3 postprocess.py

# Try tuning comparisons
tvmc run --inputs imagenet_cat.npz --output predictions.npz --print-time --repeat 1000 resnet50i1.tar 
tvmc run --inputs imagenet_cat.npz --output predictions.npz --print-time --repeat 1000 resnet50i1-tuned.tar 
