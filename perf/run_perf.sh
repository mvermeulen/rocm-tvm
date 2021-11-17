#!/bin/bash
set -x
SAVED_MODELS=${SAVED_MODELS:="/home/mev/source/rocm-migraphx/saved-models"}
TEST_RESULTDIR=${TEST_RESULTDIR:="/home/mev/source/rocm-migraphx/test-results"}

cd ${TEST_RESULTDIR}
testdir=tvm-`date '+%Y-%m-%d-%H-%M'`
mkdir $testdir
cd $testdir
pushd /src/tvm
git log | head -5 > $testdir/commit.txt
popd
while read tag batch savefile extra
do
    if [ "$tag" == "#" ]; then
	continue
    fi
    compilefile=${tag}.tar
    compilefilelibs=${tag}.libs.tar
    compilefile_nchw=${tag}.tar
    compilefilelibs_nchw=${tag}.libs.tar
    compilefile_nhwc=${tag}.tar
    compilefilelibs_nhwc=${tag}.libs.tar    
    
    tvmc -v compile --target rocm --output $compilefile ${SAVED_MODELS}/$savefile 1>$tag.compile.out 2>$tag.compile.err
    tvmc -v compile --target "rocm -libs=miopen,rocblas" --output $compilefilelibs ${SAVED_MODELS}/$savefile 1>$tag.compile.libs.out 2>$tag.compile.libs.err

    tvmc -v compile --desired-layout NCHW --target rocm --output $compilefile_nchw ${SAVED_MODELS}/$savefile 1>$tag.nchw.compile.out 2>$tag.nchw.compile.err
    tvmc -v compile --desired-layout NCHW --target "rocm -libs=miopen,rocblas" --output $compilefilelibs_nhwc ${SAVED_MODELS}/$savefile 1>$tag.nchw.compile.libs.out 2>$tag.compile.nchw.libs.err
    tvmc -v compile --desired-layout NHWC --target rocm --output $compilefile_nhwc ${SAVED_MODELS}/$savefile 1>$tag.nhwc.compile.out 2>$tag.nhwc.compile.err
    tvmc -v compile --desired-layout NHWC --target "rocm -libs=miopen,rocblas" --output $compilefilelibs_nhwc ${SAVED_MODELS}/$savefile 1>$tag.nhwc.compile.libs.out 2>$tag.nhwc.compile.libs.err    
    
    tvmc -v run --device rocm --fill-mode random --print-time --repeat 100 $compilefile 1>$tag.run.out 2>$tag.run.err
    tvmc -v run --device rocm --fill-mode random --print-time --repeat 100 $compilefilelibs 1>$tag.run.libs.out 2>$tag.run.libs.err

    tvmc -v run --device rocm --fill-mode random --print-time --repeat 100 $compilefile_nchw 1>$tag.nchw.run.out 2>$tag.nchw.run.err
    tvmc -v run --device rocm --fill-mode random --print-time --repeat 100 $compilefilelibs_nchw 1>$tag.nchw.run.libs.out 2>$tag.nchw.run.libs.err
    tvmc -v run --device rocm --fill-mode random --print-time --repeat 100 $compilefile_nhwc 1>$tag.nhwc.run.out 2>$tag.nhwc.run.err
    tvmc -v run --device rocm --fill-mode random --print-time --repeat 100 $compilefilelibs_nhwc 1>$tag.nhwc.run.libs.out 2>$tag.nhwc.run.libs.err
    
    runtime=`head -3 $tag.run.out | tail -n 1 | awk '{ print $1 }'`
    runtimelibs=`head -3 $tag.run.libs.out | tail -n 1 | awk '{ print $1 }'`

    runtime_nchw=`head -3 $tag.nchw.run.out | tail -n 1 | awk '{ print $1 }'`
    runtimelibs_nchw=`head -3 $tag.nchw.run.libs.out | tail -n 1 | awk '{ print $1 }'`
    runtime_nhwc=`head -3 $tag.nhwc.run.out | tail -n 1 | awk '{ print $1 }'`
    runtimelibs_nhwc=`head -3 $tag.nhwc.run.libs.out | tail -n 1 | awk '{ print $1 }'`
    
    echo $tag,$runtime | tee -a results.csv
    echo $tag,$runtimelibs | tee -a results.libs.csv
    echo $tag,$runtime_nchw | tee -a results.nchw.csv
    echo $tag,$runtimelibs_nchw | tee -a results.nchw.libs.csv
    echo $tag,$runtime_nhwc | tee -a results.nhwc.csv
    echo $tag,$runtimelibs_nhwc | tee -a results.nhwc.libs.csv
    echo $tag,$runtime,$runtimelibs,$runtime_nchw,$runtimelibs_nchw,$runtime_nhwc,$runtimelibs_nhwc | tee -a results.all.csv
done <<MODELLIST
torchvision-resnet50_1         1 torchvision/resnet50i1.onnx
torchvision-inceptionv3_1      1 torchvision/inceptioni1.onnx
torchvision-vgg16_1            1 torchvision/vgg16i1.onnx
cadene-dpn92_1                 1 cadene/dpn92i1.onnx
cadene-resnext101_1            1 cadene/resnext101_64x4di1.onnx
slim-vgg16_1                   1 slim/vgg16_i1.pb
slim-mobilenet_1               1 slim/mobilenet_i1.pb
slim-inceptionv4_1             1 slim/inceptionv4_i1.pb
bert-mrpc1		       1 huggingface-transformers/bert_mrpc1.onnx
MODELLIST

