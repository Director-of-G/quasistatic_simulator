# -*- mode: cmake -*-
# vi: set ft=cmake :

# Copyright (c) 2017, Massachusetts Institute of Technology.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.10.2)
project(quasistatic_simulator)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# N.B. This is a temporary flag. It only really applies to Linux, as Mac
# does not need X11.
option(RUN_X11_TESTS "Run tests that require X11" OFF)

include(CTest)

find_package(drake CONFIG REQUIRED)

find_package(PythonInterp ${drake_PYTHON_VERSION} MODULE REQUIRED EXACT)

find_package(pinocchio REQUIRED)

find_package(Eigen3 REQUIRED)

# find_package(hpp-fcl REQUIRED)

# Mosek solver
set(MOSEK_INSTALL_DIR "$ENV{MSK_HOME}/mosek/10.1/tools/platform/linux64x86")
set(MOSEK_INCLUDE_DIRS "${MOSEK_INSTALL_DIR}/h")
set(MOSEK_LIBRARIES "${MOSEK_INSTALL_DIR}/bin")
set(MOSEK_LIB_NAME "mosek64" "fusion64")

find_program(PYTHON_EXECUTABLE NAMES python3
        PATHS "${FIND_PYTHON_EXECUTABLE_PATHS}"
        NO_DEFAULT_PATH
        )

execute_process(COMMAND ${PYTHON_EXECUTABLE}-config --exec-prefix
        OUTPUT_VARIABLE PYTHON_EXEC_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
list(APPEND CMAKE_PREFIX_PATH "${PYTHON_EXEC_PREFIX}")
find_package(PythonLibs ${drake_PYTHON_VERSION} MODULE REQUIRED EXACT)

get_filename_component(PYTHONPATH
        "${drake_DIR}/../../python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages"
        REALPATH
        )
get_filename_component(PYTHONPATH
        "/opt/openrobots/lib/python3.8/site-packages"
        REALPATH
        )

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# My packages.
find_package(pybind11 CONFIG REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(GTest REQUIRED)

add_subdirectory(qsim)
add_subdirectory(diffcp)
add_subdirectory(bindings)
add_subdirectory(controllers)
# add_subdirectory(tests)
# add_subdirectory(examples)
# add_subdirectory(tests_yongpeng)

# install(EXPORT ${CMAKE_PROJECT_NAME}
#         DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
#         NAMESPACE ${CMAKE_PROJECT_NAME}::
#         FILE ${CMAKE_PROJECT_NAME}Config.cmake
# )

# install(DIRECTORY qsim diffcp
#         DESTINATION include
#         FILES_MATCHING
#         PATTERN *.h
# )
