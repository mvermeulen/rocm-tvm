#!/bin/bash
#
# Build LLVM locally rather than in the docker image.

LLVMBUILD=${LLVMBUILD:="llvmbuild"}
# needs to be local so it can be copied in...
LLVMINSTALL=${LLVMINSTALL:="${LLVMBUILD}/install"}
LLVMSRC=${LLVSRC:="${LLVMBUILD}/llvm-project"}

if [ -d ${LLVMINSTALL} ]; then
    echo ${LLVMINSTALL} already built
    exit 0
fi

if [ ! -d ${LLVMBUILD} ]; then
    mkdir ${LLVMBUILD}
fi

if [ -d ${LLVMBUILD}/llvm-project ]; then
    cd ${LLVMBUILD}/llvm-project
    git pull
else
    cd ${LLVMBUILD}
    git clone https://github.com/llvm/llvm-project.git
    git checkout llvmorg-9.0.0
    cd llvm-project
fi

if [ ! -d build ]; then
    mkdir build
    cd build
    cmake -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS="lld;compiler-rt;clang" -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" -DCMAKE_INSTALL_PREFIX=../../install ../llvm
else
    cd build
fi

make 2>&1 | tee make.log
make install 2>&1 | tee make_install.log
