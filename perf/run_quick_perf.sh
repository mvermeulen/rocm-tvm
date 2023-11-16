#!/bin/bash

MODEL=${MODEL:="/home/mev/source/rocm-migraphx/saved-models/torchvision/resnet50i1.onnx"}
model_name=`basename $MODEL .onnx`

models=(
    '/home/mev/source/rocm-migraphx/saved-models/torchvision/resnet50i1.onnx'
    '/home/mev/source/rocm-migraphx/saved-models/torchvision/resnet50i64.onnx'
    '/home/mev/source/rocm-migraphx/saved-models/onnx-model-zoo/gpt2-10.onnx'
    '/home/mev/source/rocm-migraphx/saved-models/onnx-model-zoo/inception-v2-9.onnx'
    '/home/mev/source/rocm-migraphx/saved-models/onnx-model-zoo/bertsquad-12.onnx'
)
labels=('cpu' 'rocm' 'hip')
targets=('llvm' 'rocm' 'hip')
devices=('cpu' 'rocm' 'rocm')

for full_model in ${models[*]}
do
    model=`basename $full_model .onnx`
    for i in "${!targets[@]}"
    do
	label=${labels[$i]}
	target=${targets[$i]}
	device=${devices[$i]}
	
	echo "\n--- Model: $model,  Device: $device, Target: $target"
	
	echo tvmc compile --target $target -o ${model}.${target}.tar $full_model
	tvmc compile --target $target -o ${model}.${target}.tar $full_model \
	     1> ${model}.${label}.compile.out 2> ${model}.${label}.compile.err
	
	echo tvmc run --device $device --fill-mode random --print-time --repeat 100 ${model}.${target}.tar
	tvmc run --device $device --fill-mode random --print-time --repeat 100 ${model}.${target}.tar \
	     1> ${model}.${label}.run.out 2> ${model}.${label}.run.err
	
	echo tvmc tune --target $target --output ${model}.${target}.json $full_model
	tvmc tune --target $target --output ${model}.${target}.json $full_model \
	     1> ${model}.${label}.tune.out 2>${model}.${label}.tune.err
	
	echo tvmc compile --target $target -o ${model}.${target}.tuned.tar --tuning-records ${model}.${target}.json $full_model
	tvmc compile --target $target -o ${model}.${target}.tuned.tar --tuning-records ${model}.${target}.json $full_model \
	     1> ${model}.${label}.compilet.out 2> ${model}.${label}.compilet.err
	
	echo tvmc run --device $device --fill-mode random --print-time --repeat 100 ${model}.${target}.tuned.tar
	tvmc run --device $device --fill-mode random --print-time --repeat 100 ${model}.${target}.tuned.tar \
	     1> ${model}.${label}.runt.out 2> ${model}.${label}.runt.err
    done    
done
