#!/usr/bin/env bash
set -euxo pipefail

# install pinocchio from robotpkg
apt install -qqy lsb-release curl
mkdir -p /etc/apt/keyrings
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
    | tee /etc/apt/keyrings/robotpkg.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
    | tee /etc/apt/sources.list.d/robotpkg.list
apt update
# this will install pinocchio and compatible hpp-fcl
apt install -qqy robotpkg-py3*-pinocchio

# environment
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig
echo "export PATH=/opt/openrobots/bin:$PATH" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH" >> ~/.bashrc
echo "export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH" >> ~/.bashrc
