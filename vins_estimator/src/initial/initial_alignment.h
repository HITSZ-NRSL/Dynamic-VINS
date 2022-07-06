/*
 * @Author: Jianheng Liu
 * @Date: 2022-01-11 10:50:29
 * @LastEditors: Jianheng Liu
 * @LastEditTime: 2022-01-16 11:35:13
 * @Description: Description
 */
#pragma once
#include "../factor/imu_factor.h"
#include "../feature_manager/feature_manager.h"
#include "../utility/utility.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <ros/ros.h>

using namespace Eigen;
using namespace std;

class ImageFrame
{
public:
    ImageFrame(){};
    ImageFrame(const map<int, Eigen::Matrix<double, 7, 1>> &_points, double _t)
        : t{_t}, is_key_frame{false}
    {
        points = _points;
    };
    map<int, Eigen::Matrix<double, 7, 1>> points;
    double                                t;
    Matrix3d                              R;
    Vector3d                              T;
    IntegrationBase                      *pre_integration;
    bool                                  is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g,
                        VectorXd &x);
bool LinearAlignmentWithDepth(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x);
bool LinearAlignmentWithDepthGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g,
                                     VectorXd &x);
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs);
