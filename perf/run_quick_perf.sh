#!/bin/bash

DATESTAMP=`date '+%Y%m%d-%H%M'`
mkdir ${DATESTAMP}
cd ${DATESTAMP}
RUN_TUNE=${RUN_TUNE:="0"}

models=(
    '/home/mev/source/rocm-migraphx/saved-models/torchvision/resnet50i1.onnx'
    '/home/mev/source/rocm-migraphx/saved-models/torchvision/resnet50i64.onnx'
    '/home/mev/source/rocm-migraphx/saved-models/onnx-model-zoo/inception-v2-9.onnx'
)
labels=('cpu' 'rocm' 'hip' 'rocmlib' 'nvptx' 'cuda' 'nvptxlib' 'cudalib' 'vulkan')
targets=('llvm' 'rocm' 'hip' 'rocm -libs=miopen,rocblas' 'nvptx' 'cuda' 'nvptx -libs=cudnn,cublas' 'cuda -libs=cudnn,cublas' 'vulkan')
devices=('cpu' 'rocm' 'rocm' 'rocm' 'cuda' 'cuda' 'cuda' 'cuda' 'vulkan')

for full_model in ${models[*]}
do
    model=`basename $full_model .onnx`
    for i in "${!targets[@]}"
    do
	label=${labels[$i]}
	target=${targets[$i]}
	device=${devices[$i]}

	if [ ! -d $model ];then
	    mkdir $model
	fi
	
	cd $model
	
	echo "\n--- Model: $model,  Device: $device, Target: $target"
	
	echo tvmc compile --target $target -o ${model}.${label}.tar $full_model
	tvmc compile --target "$target" -o ${model}.${label}.tar $full_model \
	     1> ${model}.${label}.compile.out 2> ${model}.${label}.compile.err
	
	echo tvmc run --device $device --fill-mode random --print-time --repeat 100 ${model}.${label}.tar
	tvmc run --device $device --fill-mode random --print-time --repeat 100 ${model}.${label}.tar \
	     1> ${model}.${label}.run.out 2> ${model}.${label}.run.err

	if [ "$RUN_TUNE" = "1" ]; then
	
	    echo tvmc tune --target $target --output ${model}.${label}.json $full_model
	    tvmc tune --target "$target" --output ${model}.${label}.json $full_model \
		 1> ${model}.${label}.tune.out 2>${model}.${label}.tune.err
	
	    echo tvmc compile --target "$target" -o ${model}.${label}.tuned.tar --tuning-records ${model}.${label}.json $full_model
	    tvmc compile --target "$target" -o ${model}.${label}.tuned.tar --tuning-records ${model}.${label}.json $full_model \
		 1> ${model}.${label}.compilet.out 2> ${model}.${label}.compilet.err
	
	    echo tvmc run --device $device --fill-mode random --print-time --repeat 100 ${model}.${label}.tuned.tar
	    tvmc run --device $device --fill-mode random --print-time --repeat 100 ${model}.${label}.tuned.tar \
		 1> ${model}.${label}.runt.out 2> ${model}.${label}.runt.err

	fi

	cd ..

    done    
done
