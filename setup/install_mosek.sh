#!/usr/bin/env sh
set -euxo pipefail

# install mosek fusion c++ API
wget https://download.mosek.com/stable/10.1.22/mosektoolslinux64x86.tar.bz2

# this creates a mosek folder in /home/mosek
tar -jxvf mosektoolslinux64x86.tar.bz2 -C /home
rm mosektoolslinux64x86.tar.bz2

# compile fusion cxx API
export MSK_HOME=/home
export PATH=$MSK_HOME/mosek/10.1/tools/platform/linux64x86/bin:$PATH
cd $MSK_HOME/mosek/10.1/tools/platform/linux64x86/src/fusion_cxx
make install

# environment
echo "export MSK_HOME=/home" >> ~/.bashrc
echo "export PATH=$MSK_HOME/mosek/10.1/tools/platform/linux64x86/bin:$PATH" >> ~/.bashrc
