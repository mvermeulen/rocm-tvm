#!/bin/bash
set -x
SAVED_MODELS=${SAVED_MODELS:="/home/mev/source/rocm-migraphx/saved-models"}
DO_RUN=${DO_RUN:="1"}

if [ ! -d tunedb_gpu ]; then
    mkdir tunedb_gpu
fi

if [ ! -d tunedb_cpu ]; then
    mkdir tunedb_cpu
fi

if [ ! -d compiledb_gpu ]; then
    mkdir compiledb_gpu
fi

if [ ! -d compiledb_cpu ]; then
    mkdir compiledb_cpu
fi

if [ ! -d profiledb_gpu ]; then
    mkdir profiledb_gpu
fi

if [ ! -d profiledb_cpu ]; then
    mkdir profiledb_cpu
fi

if [ ! -d rundb_gpu ]; then
    mkdir rundb_gpu
fi

if [ ! -d rundb_cpu ]; then
    mkdir rundb_cpu
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

    # tune for the GPU
    tunedb_gpu_autotvm=tunedb_gpu/${model}.autotvm.json
    tunedb_gpu_autoscheduler=tunedb_gpu/${model}.autosheduler.json
    if [ ! -f $tunedb_gpu_autotvm ]; then
	echo "Tuning $tunedb_gpu_autotvm ---" | tee -a $logfile
	tvmc -v tune --target "rocm" --output $tunedb_gpu_autotvm $onnxfile 2>&1 | tee -a $logfile	
    fi
    if [ ! -f $tunedb_gpu_autoscheduler ]; then
	echo "Tuning $tunedb_gpu_autoscheduler ---" | tee -a $logfile
	tvmc -v tune --target "rocm" --output $tunedb_gpu_autoscheduler --enable-autoscheduler $onnxfile 2>&1 | tee -a $logfile		
    fi    

    # tune for the CPU
    tunedb_cpu_autotvm=tunedb_cpu/${model}.autotvm.json
    tunedb_cpu_autoscheduler=tunedb_cpu/${model}.autosheduler.json
    if [ ! -f $tunedb_cpu_autotvm ]; then
	echo "Tuning $tunedb_cpu_autotvm ---" | tee -a $logfile
	tvmc -v tune --target "llvm" --output $tunedb_cpu_autotvm $onnxfile 2>&1 | tee -a $logfile	
    fi
    if [ ! -f $tunedb_cpu_autoscheduler ]; then
	echo "Tuning $tunedb_cpu_autoscheduler ---" | tee -a $logfile
	tvmc -v tune --target "llvm" --output $tunedb_cpu_autoscheduler --enable-autoscheduler $onnxfile 2>&1 | tee -a $logfile		
    fi

    # compile for the GPU
    compiledb_gpu_untuned=compiledb_gpu/${model}.untuned.tar
    compiledb_gpu_autotvm=compiledb_gpu/${model}.autotvm.tar
    compiledb_gpu_autoscheduler=compiledb_gpu/${model}.autoscheduler.tar

    if [ ! -f $compiledb_gpu_untuned ]; then
	echo "Compiling $compiledb_gpu_untuned ---" | tee -a $logfile
	tvmc -v compile --target "rocm" --output $compiledb_gpu_untuned $onnxfile 2>&1 | tee -a $logfile
    fi
    if [ ! -f $compiledb_gpu_autotvm ]; then
	echo "Compiling $compiledb_gpu_autotvm ---" | tee -a $logfile	
	tvmc -v compile --target "rocm" --output $compiledb_gpu_autotvm --tuning-records $tunedb_gpu_autotvm $onnxfile 2>&1 | tee -a $logfile
    fi
    if [ ! -f $compiledb_gpu_autoscheduler ]; then
	echo "Compiling $compiledb_gpu_autoscheduler ---" | tee -a $logfile	
	tvmc -v compile --target "rocm" --output $compiledb_gpu_autoscheduler --tuning-records $tunedb_gpu_autoscheduler $onnxfile 2>&1 | tee -a $logfile
    fi

    # compile for the CPU
    compiledb_cpu_untuned=compiledb_cpu/${model}.untuned.tar
    compiledb_cpu_autotvm=compiledb_cpu/${model}.autotvm.tar
    compiledb_cpu_autoscheduler=compiledb_cpu/${model}.autoscheduler.tar

    if [ ! -f $compiledb_cpu_untuned ]; then
	echo "Compiling $compiledb_cpu_untuned ---" | tee -a $logfile
	tvmc -v compile --target "llvm" --output $compiledb_cpu_untuned $onnxfile 2>&1 | tee -a $logfile
    fi
    if [ ! -f $compiledb_cpu_autotvm ]; then
	echo "Compiling $compiledb_cpu_autotvm ---" | tee -a $logfile	
	tvmc -v compile --target "llvm" --output $compiledb_cpu_autotvm --tuning-records $tunedb_cpu_autotvm $onnxfile 2>&1 | tee -a $logfile
    fi
    if [ ! -f $compiledb_cpu_autoscheduler ]; then
	echo "Compiling $compiledb_cpu_autoscheduler ---" | tee -a $logfile	
	tvmc -v compile --target "llvm" --output $compiledb_cpu_autoscheduler --tuning-records $tunedb_cpu_autoscheduler $onnxfile 2>&1 | tee -a $logfile
    fi

    # gather profile information for GPU
    profiledb_gpu_untuned=profiledb_gpu/${model}.untuned.txt
    profiledb_gpu_autotvm=profiledb_gpu/${model}.autotvm.txt    
    profiledb_gpu_autoscheduler=profiledb_gpu/${model}.autoscheduler.txt

    if [ ! -f $profiledb_gpu_untuned ]; then
	echo "Profiling $profiledb_gpu_untuned --" | tee -a $logfile
	tvmc -v run --device "rocm" --profile --file-mode random --print-time --repeat 100 $compiledb_gpu_untuned > $profiledb_gpu_untuned
    fi
    if [ ! -f $profiledb_gpu_autotvm ]; then
	echo "Profiling $profiledb_gpu_autotvm --" | tee -a $logfile
	tvmc -v run --device "rocm" --profile --file-mode random --print-time --repeat 100 $compiledb_gpu_autotvm > $profiledb_gpu_autotvm
    fi
    if [ ! -f $profiledb_gpu_autoscheduler ]; then
	echo "Profiling $profiledb_gpu_autoscheduler --" | tee -a $logfile
	tvmc -v run --device "rocm" --profile --file-mode random --print-time --repeat 100 $compiledb_gpu_autoscheduler > $profiledb_gpu_autoscheduler
    fi

    # gather profile information for CPU
    profiledb_cpu_untuned=profiledb_cpu/${model}.untuned.txt
    profiledb_cpu_autotvm=profiledb_cpu/${model}.autotvm.txt    
    profiledb_cpu_autoscheduler=profiledb_cpu/${model}.autoscheduler.txt

    if [ ! -f $profiledb_cpu_untuned ]; then
	echo "Profiling $profiledb_cpu_untuned --" | tee -a $logfile
	tvmc -v run --device "cpu" --profile --file-mode random --print-time --repeat 100 $compiledb_cpu_untuned > $profiledb_cpu_untuned
    fi
    if [ ! -f $profiledb_cpu_autotvm ]; then
	echo "Profiling $profiledb_cpu_autotvm --" | tee -a $logfile
	tvmc -v run --device "cpu" --profile --file-mode random --print-time --repeat 100 $compiledb_cpu_autotvm > $profiledb_cpu_autotvm
    fi
    if [ ! -f $profiledb_cpu_autoscheduler ]; then
	echo "Profiling $profiledb_cpu_autoscheduler --" | tee -a $logfile
	tvmc -v run --device "cpu" --profile --file-mode random --print-time --repeat 100 $compiledb_cpu_autoscheduler > $profiledb_cpu_autoscheduler
    fi

    # run the GPU models
    rundb_gpu_untuned=rundb_gpu/${model}.untuned.txt
    rundb_gpu_autotvm=rundb_gpu/${model}.autotvm.txt
    rundb_gpu_autoscheduler=rundb_gpu/${model}.autoscheduler.txt

    if [ ! -f $rundb_gpu_untuned ]; then
	echo "Profiling $profile_gpu_untuned --" | tee -a $logfile
	tvmc -v run --device "rocm" --profile --file-mode random --print-time --repeat 100 $compiledb_gpu_untuned > $rundb_gpu_untuned
    fi
    if [ ! -f $rundb_gpu_autotvm ]; then
	echo "Profiling $rundb_gpu_autotvm --" | tee -a $logfile
	tvmc -v run --device "rocm" --profile --file-mode random --print-time --repeat 100 $compiledb_gpu_autotvm > $rundb_gpu_autotvm
    fi
    if [ ! -f $rundb_gpu_autoscheduler ]; then
	echo "Profiling $rundb_gpu_autoscheduler --" | tee -a $logfile
	tvmc -v run --device "rocm" --profile --file-mode random --print-time --repeat 100 $compiledb_gpu_autoscheduler > $rundb_gpu_autoscheduler
    fi

    # run the CPU models
    rundb_cpu_untuned=rundb_cpu/${model}.untuned.txt
    rundb_cpu_autotvm=rundb_cpu/${model}.autotvm.txt
    rundb_cpu_autoscheduler=rundb_cpu/${model}.autoscheduler.txt

    if [ ! -f $rundb_cpu_untuned ]; then
	echo "Profiling $profile_cpu_untuned --" | tee -a $logfile
	tvmc -v run --device "rocm" --profile --file-mode random --print-time --repeat 100 $compiledb_cpu_untuned > $rundb_cpu_untuned
    fi
    if [ ! -f $rundb_cpu_autotvm ]; then
	echo "Profiling $rundb_cpu_autotvm --" | tee -a $logfile
	tvmc -v run --device "rocm" --profile --file-mode random --print-time --repeat 100 $compiledb_cpu_autotvm > $rundb_cpu_autotvm
    fi
    if [ ! -f $rundb_cpu_autoscheduler ]; then
	echo "Profiling $rundb_cpu_autoscheduler --" | tee -a $logfile
	tvmc -v run --device "rocm" --profile --file-mode random --print-time --repeat 100 $compiledb_cpu_autoscheduler > $rundb_cpu_autoscheduler
    fi    
    
done <<MODELLIST
torchvision-resnet50i1  torchvision/resnet50i1.onnx
torchvision-inceptioni1 torchvision/inceptioni1.onnx
torchvision-vgg16       torchvision/vgg16i1.onnx
MODELLIST
