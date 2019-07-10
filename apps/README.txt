Notes about apps when run on ROCm.

The apps/benchmark/gpu_imagenet_bench.py script did not produce output.
After some helpful feedback from TVM community forum:
https://discuss.tvm.ai/t/rocm-apps-benchmark-gpu-imagenet-py-failures-loading-kernel/3218/9

Following is a better description of what runs:
1. Modify the script to no longer add --model=gfx900

    if args.target is not 'rocm':
       target = tvm.target.create('%s -model=%s' % (args.target, args.model))
    else:
       target = args.target

2. After this the following was the status:

   * resnet-18       - core dump
   * resnet-34       - core dump
   * resnet-50       - OK
   * vgg-16          - OK
   * vgg-19          - OK
   * densenet-121    - core dump
   * inception_v3    - OK
   * mobilenet	     - OK
   * mobilenet_v2    - core dump
   * squeezenet_v1.0 - OK
   * squeezenet_v1.1 - OK
