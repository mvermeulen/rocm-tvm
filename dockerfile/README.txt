# Script to build docker containers with different versions of ROCm

    ./build_dockerfile.sh

as well as checked in versions of some Dockerfiles.

These scripts take care of some identified issues:
* Prebuilt copies of LLVM had problems including [Issue 3058 at TVM repository](https://github.com/dmlc/tvm/issues/3058), so the docker container builds LLVM from source.
* Changed the code metadata attribute as suggested in [Issue 3058 at TVM repository](https://github.com/dmlc/tvm/issues/3058).