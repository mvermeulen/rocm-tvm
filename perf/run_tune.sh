#!/bin/bash
set -x
SAVED_MODELS=${SAVED_MODELS:="/home/mev/source/rocm-migraphx/saved-models"}
DO_RUN=${DO_RUN:="1"}

if [ ! -d compiledb_gpu_tuned ]; then
    mkdir compiledb_gpu_tuned
fi

if [ ! -d compiledb_gpu_untuned ]; then
    mkdir compiledb_gpu_untuned
fi

if [ ! -d compiledb_cpu_tuned ]; then
    mkdir compiledb_cpu_tuned
fi

if [ ! -d compiledb_cpu_untuned ]; then
    mkdir compiledb_cpu_untuned
fi

if [ ! -d tunedb_gpu ]; then
    mkdir tunedb_gpu
fi

if [ ! -d tunedb_cpu ]; then
    mkdir tunedb_cpu
fi

if [ ! -d log ]; then
    mkdir log
fi

while read model path
do
    # make sure model is present
    onnxfile=${SAVED_MODELS}/$path
    if [ ! -f $onnxfile ]; then
	echo $onnxfile is missing
	exit 1
    fi

    logfile=log/${model}.log

    # tuning for gpu
    tunedb_gpu=tunedb_gpu/${model}.json
    if [ ! -f $tunedb_gpu ]; then
	echo "Tuning $tunedb_gpu ---" | tee -a $logfile	
	tvmc -v tune --target "rocm" --output $tunedb_gpu $onnxfile 2>&1 | tee -a $logfile
    fi

    # tuning for cpu
    tunedb_cpu=tunedb_cpu/${model}.json
    if [ ! -f $tunedb_cpu ]; then
	echo "Tuning $tunedb_cpu ---" | tee -a $logfile	
	tvmc -v tune --target "llvm" --output $tunedb_cpu $onnxfile 2>&1 | tee -a $logfile
    fi

    # build not tuned version for gpu
    compiledb_gpu_untuned=compiledb_gpu_untuned/${model}.tar
    if [ ! -f $compiledb_gpu_untuned ]; then
	echo "Compiling $compiledb_gpu_untuned ---" | tee -a $logfile
	tvmc -v compile --target "rocm" --output $compiledb_gpu_untuned $onnxfile 2>&1 | tee -a $logfile
    fi

    # build not tuned version for cpu
    compiledb_cpu_untuned=compiledb_cpu_untuned/${model}.tar
    if [ ! -f $compiledb_cpu_untuned ]; then
	echo "Compiling $compiledb_cpu_untuned ---" | tee -a $logfile
	tvmc -v compile --target "llvm" --output $compiledb_cpu_untuned $onnxfile 2>&1 | tee -a $logfile
    fi    

    # build tuned version for gpu
    compiledb_gpu_tuned=compiledb_gpu_tuned/${model}.tar
    if [ ! -f $compiledb_gpu_tuned ]; then
	echo "Compiling $compiledb_gpu_tuned ---" | tee -a $logfile
	tvmc -v compile --target "rocm" --output $compiledb_gpu_tuned --tuning-records $tunedb_gpu $onnxfile 2>&1 | tee -a $logfile
    fi
    # build tuned version for cpu
    compiledb_cpu_tuned=compiledb_cpu_tuned/${model}.tar
    if [ ! -f $compiledb_cpu_tuned ]; then
	echo "Compiling $compiledb_cpu_tuned ---" | tee -a $logfile
	tvmc -v compile --target "llvm" --output $compiledb_cpu_tuned --tuning-records $tunedb_cpu $onnxfile 2>&1 | tee -a $logfile
    fi
    
    # run the model step
    if [ "${DO_RUN}" = "0" ]; then
	continue
    fi
    echo "Running $model ---" | tee -a $logfile		    
    tvmc -v run --device "rocm" --fill-mode random --print-time --repeat 100 $compiledb_gpu_tuned 2>&1 | tee -a $logfile
    tvmc -v run --device "rocm" --fill-mode random --print-time --repeat 100 $compiledb_gpu_untuned 2>&1 | tee -a $logfile
    tvmc -v run --device "cpu" --fill-mode random --print-time --repeat 100 $compiledb_cpu_tuned 2>&1 | tee -a $logfile
    tvmc -v run --device "cpu" --fill-mode random --print-time --repeat 100 $compiledb_cpu_untuned 2>&1 | tee -a $logfile
done <<MODELLIST
torchvision-resnet50i1  torchvision/resnet50i1.onnx
torchvision-inceptioni1 torchvision/inceptioni1.onnx
torchvision-vgg16       torchvision/vgg16i1.onnx
MODELLIST
