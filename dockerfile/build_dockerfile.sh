#!/bin/bash
#
# This script asks configuration information to build Dockerfiles
# that build TVM for different configurations including ROCm releases.
#

# Name of output dockerfile can be passed in as an environment variable.
DOCKERFILE=${DOCKERFILE:="Dockerfile"}
if [ -f ${DOCKERFILE} ]; then
    echo File ${DOCKERFILE} exists, please move out of way before running this script.
    exit 1
fi

# The base package is dev-ubuntu-<version>:<rocm>, ask following questions to configure
read -p "Enter ROCm version [3.3]: " rocm_version
rocm_version=${rocm_version:="3.3"}

read -p "Enter Ubuntu version [18.04]: " ubuntu_version
ubuntu_version=${ubuntu_version:="18.04"}

echo "FROM rocm/dev-ubuntu-${ubuntu_version}:${rocm_version}" > ${DOCKERFILE}
echo "RUN sed -e 's/debian/${rocm_version}/g' /etc/apt/sources.list.d/rocm.list > /etc/apt/sources.list.d/rocm${rocm_version}.list" >> ${DOCKERFILE}
echo "RUN rm /etc/apt/sources.list.d/rocm.list" >> ${DOCKERFILE}

# Install prerequisite packages (llvm build, ROCm runtime, TVM build)
echo "ENV PATH=/usr/local/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" >> ${DOCKERFILE}
echo "ENV DEBIAN_FRONTEND=noninteractive" >> ${DOCKERFILE}
echo "RUN apt update && apt install -y git wget libz3-dev libxml2-dev openssl" >> ${DOCKERFILE}
echo "RUN cd /src && wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz && tar xf cmake-3.17.3.tar.gz && cd cmake-3.17.3 && ./configure && make && make install" >> ${DOCKERFILE}
echo "RUN apt update && apt install -y rocm-libs miopen-hip" >> ${DOCKERFILE}
echo "RUN apt update && apt install -y python python-dev python-setuptools gcc libtinfo-dev zlib1g-dev build-essential python3 python3-pip" >> ${DOCKERFILE}

echo "RUN mkdir /src" >> ${DOCKERFILE}
# AMDGPU version of LLVM including lld.ld is required to run TVM.
# Seems to be some issues with prebuilt packages, so build from source.
read -p "Copy LLVM to docker? [Y]: " copy_llvm
copy_llvm=${copy_llvm:="Y"}
if [ "${copy_llvm}" == 'Y' -o "${copy_llvm}" == 'y' ]; then
    env LLVMBUILD="llvmbuild9" ./build_llvm9.sh
    echo "COPY llvmbuild9/install/ /usr/local/llvm/" >> ${DOCKERFILE}
else
   echo "RUN cd /src && git clone https://github.com/llvm/llvm-project.git" >> ${DOCKERFILE}
   echo "RUN cd /src/llvm-project && mkdir build" >> ${DOCKERFILE}
   echo "RUN cd /src/llvm-project/build && cmake -G \"Unix Makefiles\" -DLLVM_ENABLE_PROJECTS=\"lld;compiler-rt;clang\" -DLLVM_TARGETS_TO_BUILD=\"X86;AMDGPU\" -DCMAKE_INSTALL_PREFIX=/usr/local/llvm ../llvm" >> ${DOCKERFILE}
   echo "RUN cd /src/llvm-project/build && make 2>&1 | tee make.log && make install" >> ${DOCKERFILE}
fi
# Build TVM
echo "RUN cd /src && git clone --recursive https://github.com/dmlc/tvm" >> ${DOCKERFILE}

# Patch source files mentioned here: https://github.com/dmlc/tvm/issues/3058
echo "RUN mkdir /src/tvm/build" >> ${DOCKERFILE}
echo "RUN cd /src/tvm/build && sed -e 's/USE_ROCM OFF/USE_ROCM ON/g' -e 's?USE_LLVM OFF?USE_LLVM /usr/local/llvm/bin/llvm-config?g' -e 's/USE_MIOPEN OFF/USE_MIOPEN ON/g' -e 's/USE_ROCBLAS OFF/USE_ROCBLAS ON/g' ../cmake/config.cmake > config.cmake" >> ${DOCKERFILE}
echo "RUN cd /src/tvm/build && cmake .. && make" >> ${DOCKERFILE}
echo "RUN cd /src/tvm/python && python3 setup.py install" >> ${DOCKERFILE}
echo "RUN cd /src/tvm/topi/python && python3 setup.py install" >> ${DOCKERFILE}
echo "RUN cd /src && git clone https://github.com/mvermeulen/rocm-tvm" >> ${DOCKERFILE}
echo "RUN pip3 install scipy psutil xgboost tornado" >> ${DOCKERFILE}
