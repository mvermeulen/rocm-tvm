#!/bin/bash
set -x
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
    tvmc -v run --device rocm --fill-mode random --print-time --repeat 100 $compilefile 1>$tag.run.out 2>$tag.run.err
    runtime=`head -3 $tag.run.out | tail -n 1 | awk '{ print $1 }'`
    echo $tag,$runtime
    echo $tag,$runtime | tee -a results.csv
done <<MODELLIST
torchvision-resnet50_1         1 torchvision/resnet50i1.onnx
torchvision-inceptionv3_1      1 torchvision/inceptioni1.onnx
torchvision-vgg16_1            1 torchvision/vgg16i1.onnx
cadene-dpn92_1                 1 cadene/dpn92i1.onnx
cadene-resnext101_1            1 cadene/resnext101_64x4di1.onnx
slim-vgg16_1                   1 slim/vgg16_i1.pb
slim-mobilenet_1               1 slim/mobilenet_i1.pb
slim-inceptionv4_1             1 slim/inceptionv4_i1.pb
MODELLIST

