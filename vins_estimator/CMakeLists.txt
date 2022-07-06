cmake_minimum_required(VERSION 2.8.3)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "RelWithDebInfo")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        geometry_msgs
        nav_msgs
        tf
        cv_bridge
        camera_model
        message_filters
        yolo_ros
        image_transport

        nodelet
        )
        
find_package(OpenCV REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_library(vins_lib
        src/utility/parameters.cpp
        src/estimator/estimator.cpp
        src/feature_manager/feature_manager.cpp
        src/factor/pose_local_parameterization.cpp
        src/factor/projection_factor.cpp
        src/factor/projection_td_factor.cpp
        src/factor/marginalization_factor.cpp
        src/utility/utility.cpp
        src/utility/visualization.cpp
        src/utility/CameraPoseVisualization.cpp
        src/initial/solve_5pts.cpp
        src/initial/initial_aligment.cpp
        src/initial/initial_sfm.cpp
        src/initial/initial_ex_rotation.cpp
        src/feature_tracker/feature_tracker.cpp)
target_link_libraries(vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} /usr/local/lib/libSophus.so
)

add_library(estimator_nodelet src/estimator_nodelet.cpp)
target_link_libraries(estimator_nodelet vins_lib)
add_dependencies(estimator_nodelet yolo_ros_generate_messages_cpp)
