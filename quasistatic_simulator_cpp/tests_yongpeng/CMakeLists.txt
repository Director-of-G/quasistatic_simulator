cmake_minimum_required(VERSION 3.10.2)
set(CMAKE_BUILD_TYPE Debug)
project(test_yongpeng)

# Find the Eigen3 library
find_package(Eigen3 REQUIRED)

# Find the Pinocchio library
find_package(pinocchio REQUIRED)

find_package(hpp-fcl REQUIRED)
MESSAGE(STATUS "HPP_FCL_INCLUDE_DIR = ${HPP_FCL_LIBRARIES}.")

# Add your source files here
add_executable(${PROJECT_NAME} test_contact_jacobian_pinocchio.cc)  # Change main.cpp to your file

# Include directories for Eigen and Pinocchio
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${HPP_FCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PINOCCHIO_INCLUDE_DIRS})

# Link against Pinocchio
target_link_libraries(${PROJECT_NAME} ${PINOCCHIO_LIBRARIES} ${HPP_FCL_LIBRARIES})
