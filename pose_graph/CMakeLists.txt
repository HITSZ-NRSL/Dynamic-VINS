cmake_minimum_required(VERSION 2.8.3)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
# https://blog.csdn.net/qq_24624539/article/details/111056791
#set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        nav_msgs
        camera_model
        cv_bridge
        roslib
        message_filters

        nodelet
        )

find_package(OpenCV)

find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_library(pose_graph_lib
        src/pose_graph/pose_graph.cpp
        src/keyframe/keyframe.cpp
        src/utility/CameraPoseVisualization.cpp
        src/ThirdParty/DBoW/BowVector.cpp
        src/ThirdParty/DBoW/FBrief.cpp
        src/ThirdParty/DBoW/FeatureVector.cpp
        src/ThirdParty/DBoW/QueryResults.cpp
        src/ThirdParty/DBoW/ScoringObject.cpp
        src/ThirdParty/DUtils/Random.cpp
        src/ThirdParty/DUtils/Timestamp.cpp
        src/ThirdParty/DVision/BRIEF.cpp
        src/ThirdParty/VocabularyBinary.cpp
        )
target_link_libraries(pose_graph_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
#message("catkin_lib  ${catkin_LIBRARIES}")

add_library(pose_graph_nodelet src/pose_graph_nodelet.cpp)
target_link_libraries(pose_graph_nodelet pose_graph_lib)