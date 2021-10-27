#!/bin/bash
SAVED_MODELS=${SAVED_MODELS:="/home/mev/source/rocm-migraphx/saved-models"}
TEST_RESULTDIR=${TEST_RESULTDIR:="/home/mev/source/rocm-migraphx/test-results"}

cd ${TEST_RESULTDIR}
testdir=tvm-`date '+%Y-%m-%d-%H-%M'`
mkdir $testdir
cd $testdir

while read tag batch savefile extra
do
    if [ "$tag" == "#" ]; then
	continue
    fi
    compilefile=${tag}.tar
    tvmc -v compile --target rocm --output $compilefile ${SAVED_MODELS}/$savefile 1>$tag.compile.out 2>$tag.compile.err
    tvmc -v run --device rocm --fill-mode random --print-time repeat 100 $compilefile 1>$tag.run.out 2>$tag.run.err
done <<MODELLIST
torchvision-resnet50	64	torchvision/resnet50i64.onnx
MODELLIST

