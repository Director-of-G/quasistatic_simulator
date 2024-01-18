# cmake_minimum_required(VERSION 3.10.2)
# set(CMAKE_BUILD_TYPE Debug)
# project(test_yongpeng)

# # # Find the Eigen3 library
# find_package(Eigen3 REQUIRED)

# # # Find the Pinocchio library
# find_package(pinocchio REQUIRED)

# find_package(drake CONFIG REQUIRED)

# Find the Mosek library
set(MOSEK_INSTALL_DIR "/home/yongpeng/mosek-10.1/mosek/10.1/tools/platform/linux64x86")
set(MOSEK_INCLUDE_DIRS "${MOSEK_INSTALL_DIR}/h")
set(MOSEK_LIBRARIES "${MOSEK_INSTALL_DIR}/bin")
set(MOSEK_LIB_NAME "mosek64" "fusion64")

# find_package(hpp-fcl REQUIRED)
# MESSAGE(STATUS "MOSEK_LIBRARIES = ${MOSEK_LIBRARIES}.")

# Add your source files here
# add_executable(test_pinocchio test_pinocchio.cpp)  # Change main.cpp to your file
# add_executable(test_pinocchio_calculator test_pinocchio_calculator.cpp)
# add_executable(test_contact_jacobian_pinocchio test_contact_jacobian_pinocchio.cpp)
add_executable(test_mosek test_mosek_solver.cpp)

# Include directories for Eigen and Pinocchio
# target_include_directories(test_pinocchio SYSTEM PUBLIC ${HPP_FCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PINOCCHIO_INCLUDE_DIRS})
# target_include_directories(test_pinocchio_calculator PUBLIC ${PROJECT_SOURCE_DIR} ${HPP_FCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PINOCCHIO_INCLUDE_DIRS})
# target_include_directories(test_contact_jacobian_pinocchio SYSTEM PUBLIC ${HPP_FCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PINOCCHIO_INCLUDE_DIRS})
target_include_directories(test_mosek SYSTEM PUBLIC ${MOSEK_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# Link against Pinocchio
# target_link_libraries(test_pinocchio ${PINOCCHIO_LIBRARIES} ${HPP_FCL_LIBRARIES})
# target_link_libraries(test_pinocchio_calculator PUBLIC pinocchio_calculator ${EIGEN3_LIBRARIES} ${PINOCCHIO_LIBRARIES} ${HPP_FCL_LIBRARIES})
# target_link_libraries(test_contact_jacobian_pinocchio ${PINOCCHIO_LIBRARIES} ${HPP_FCL_LIBRARIES})

target_link_directories(test_mosek PUBLIC ${MOSEK_LIBRARIES})
target_link_libraries(test_mosek drake::drake ${MOSEK_LIB_NAME} ${EIGEN3_LIBRARIES})
