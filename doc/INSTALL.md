# Dynamic-VINS

## 1. Prerequisites

**ROS**
```
sudo apt-get install ros-melodic-cv-bridge ros-melodic-tf ros-melodic-message-filters ros-melodic-image-transport ros-melodic-nav-msgs ros-melodic-visualization-msgs
```

**Ceres-Solver**
```
# CMake
sudo apt-get install cmake
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev libgflags-dev
# BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse and CXSparse (optional)
sudo apt-get install libsuitesparse-dev
```
```
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout 2.0.0
mkdir ceres-bin
cd ceres-bin
cmake ..
make -j3
sudo make install
```

**Sophus**
```
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff  #版本回溯
```
`gedit sophus/so2.cpp` modify `sophus/so2.cpp` as
```
SO2::SO2()
{
  unit_complex_.real(1.0);
  unit_complex_.imag(0.0);
}
```
build
```
mkdir build && cd build && cmake .. && sudo make install
```



## 2. Prerequisites for object detection 

We offer two kinds of device for tests, please follow the instruction for your match device.

### 2.1. NVIDIA devices

Clone the repository and catkin_make:

```
cd {YOUR_WORKSPACE}/src
git clone https://github.com/HITSZ-NRSL/Dynamic-VINS.git

# yolo_ros
git clone https://github.com/jianhengLiu/yolo_ros.git
# install python dependencies
sudo apt install ros-melodic-ros-numpy
sudo apt install python3-pip
pip3 install --upgrade pip
# conda create -n dvins python=3.6
# conda activate dvins
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# if you are running on xavier, please not directly install torch by pip and refer to `Possible Problems on NVIDIA Jetson AGX Xavier` section
pip3 install -r yolo_ros/requirements.txt


# build
cd ..
catkin_make
```

**If the output frame rate of yolo is lower than your video frame rate may result in bad performance of Dynamic-VINS.**

try TensorRT acceleration version of `yolo_ros`: https://github.com/jianhengLiu/yolo_ros/tree/tensorrt


### 2.2. HUAWEI Atlas200


1. prequisities

```
  sudo apt install ros-melodic-image-transport-plugins
```

2. Clone the repository:

```
  cd {YOUR_WORKSPACE}/src
  git clone https://github.com/HITSZ-NRSL/Dynamic-VINS.git

  git clone https://github.com/jianhengLiu/compressedimg2img.git

  # yolo_ros
  git clone https://github.com/jianhengLiu/yolo_ros.git
  cd yolo_ros
  git checkout atlas200
```

2. Download pre-trained model and put it in `{YOUR_WORKSPACE}/src/yolo_ros/model` directory.
   * 百度网盘： https://pan.baidu.com/s/1m0lapSFk8KG5Z1Jo5T2VFQ  密码: wgs5
   * gitee: https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/yolov3_framework_caffe_aipp_1_batch_1_input_int8_output_FP32.om
     * After you download the model, change name `yolov3_framework_caffe_aipp_1_batch_1_input_int8_output_FP32.om` to `yolov3.om`.

3. Allocate more cpu cores for VINS 
```
  sudo npu-smi set -t aicpu-config -i 0 -c 0 -d 2
  sudo reboot
```

4.  build
```   
  cd ../..
  catkin_make
```