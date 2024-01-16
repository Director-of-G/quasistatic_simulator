# cmake_minimum_required(VERSION 3.10.2)
# set(CMAKE_BUILD_TYPE Debug)
# project(test_yongpeng)

# # Find the Eigen3 library
# find_package(Eigen3 REQUIRED)

# # Find the Pinocchio library
# find_package(pinocchio REQUIRED)

# find_package(hpp-fcl REQUIRED)
# MESSAGE(STATUS "HPP_FCL_INCLUDE_DIR = ${HPP_FCL_LIBRARIES}.")

# Add your source files here
add_executable(test_pinocchio test_pinocchio.cpp)  # Change main.cpp to your file
add_executable(test_pinocchio_calculator test_pinocchio_calculator.cpp)
add_executable(test_contact_jacobian_pinocchio test_contact_jacobian_pinocchio.cpp)

# Include directories for Eigen and Pinocchio
target_include_directories(test_pinocchio SYSTEM PUBLIC ${HPP_FCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PINOCCHIO_INCLUDE_DIRS})
target_include_directories(test_pinocchio_calculator PUBLIC ${PROJECT_SOURCE_DIR} ${HPP_FCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PINOCCHIO_INCLUDE_DIRS})
target_include_directories(test_contact_jacobian_pinocchio SYSTEM PUBLIC ${HPP_FCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PINOCCHIO_INCLUDE_DIRS})

# Link against Pinocchio
target_link_libraries(test_pinocchio ${PINOCCHIO_LIBRARIES} ${HPP_FCL_LIBRARIES})
target_link_libraries(test_pinocchio_calculator PUBLIC pinocchio_calculator ${EIGEN3_LIBRARIES} ${PINOCCHIO_LIBRARIES} ${HPP_FCL_LIBRARIES})
target_link_libraries(test_contact_jacobian_pinocchio ${PINOCCHIO_LIBRARIES} ${HPP_FCL_LIBRARIES})
