#!/bin/bash
set -x

MODEL=${MODEL:="/home/mev/source/rocm-migraphx/saved-models/torchvision/resnet50i1.onnx"}

model_name=`basename $MODEL .onnx`
echo "Model:" $model_name

tvmc compile --target llvm -o ${model_name}.cpu.tar $MODEL \
     1> ${model_name}.cpu.compile.out 2> ${model_name}.cpu.compile.err
tvmc run --device cpu --fill-mode random --print-time --repeat 100 ${model_name}.cpu.tar \
     1> ${model_name}.cpu.run.out 2> ${model_name}.cpu.run.err

tvmc compile --target rocm -o ${model_name}.rocm.tar $MODEL \
     1> ${model_name}.rocm.compile.out 2> ${model_name}.rocm.compile.err
tvmc run --device rocm --fill-mode random --print-time --repeat 100 ${model_name}.rocm.tar \
     1> ${model_name}.rocm.run.out 2> ${model_name}.rocm.run.err

tvmc compile --target hip -o ${model_name}.hip.tar $MODEL \
     1> ${model_name}.hip.compile.out 2> ${model_name}.hip.compile.err
tvmc run --device rocm --fill-mode random --print-time --repeat 100 ${model_name}.hip.tar \
     1> ${model_name}.hip.run.out 2> ${model_name}.hip.run.err

tvmc tune --target llvm --output ${model_name}.cpu.tunerecords.json $MODEL \
     1> ${model_name}.cpu.tune.out 2> ${model_name}.cpu.tune.err
tvmc compile --target llvm -o ${model_name}.cputuned.tar $MODEL --tuning-records ${model_name}.cpu.tunerecords.json \
     1> ${model_name}.cpu.compiletuned.out 2> ${model_name}.cpu.compiletuned.err
tvmc run --device cpu --fill-mode random --print-time --repeat 100 ${model_name}.cputuned.tar \
     1> ${model_name}.cpu.runtuned.out 2> ${model_name}.cpu.runtuned.err

tvmc tune --target hip --output ${model_name}.hip.tunerecords.json $MODEL \
     1> ${model_name}.hip.tune.out 2> ${model_name}.hip.tune.err
tvmc compile --target hip -o ${model_name}.hiptuned.tar $MODEL --tuning-records ${model_name}.hip.tunerecords.json \
     1> ${model_name}.hip.compiletuned.out 2> ${model_name}.hip.compiletuned.err
tvmc run --device rocm --fill-mode random --print-time --repeat 100 ${model_name}.hiptuned.tar \
     1> ${model_name}.hip.runtuned.out 2> ${model_name}.hip.runtuned.err
