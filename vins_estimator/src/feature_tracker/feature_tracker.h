#pragma once

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <iostream>
#include <queue>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "../utility/parameters.h"
#include "../utility/tic_toc.h"

#include "ThreadPool.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);

void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();

    void readImage(const cv::Mat &_img, double _cur_time,
                   const Matrix3d &_relative_R = Matrix3d::Identity());

    void setMask();

    void addPoints();

    void addPoints(vector<cv::KeyPoint> &Keypts);

    void addPoints(int n_max_cnt, vector<cv::KeyPoint> &Keypts);

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    void predictPtsInNextFrame(const Matrix3d &_relative_R);

    Eigen::Vector3d get3dPt(const cv::Mat &_depth, const cv::Point2f &pt);

    void initGridsDetector();

    static bool inBorder(const cv::Point2f &pt);

    std::vector<cv::KeyPoint> gridDetect(size_t grid_id);

    cv::Mat               mask;
    cv::Mat               fisheye_mask;
    cv::Mat               cur_img, forw_img;
    vector<cv::Point2f>   n_pts;
    vector<cv::Point2f>   cur_pts, forw_pts, predict_pts, unstable_pts;
    vector<cv::Point2f>   prev_un_pts, cur_un_pts;
    vector<cv::Point2f>   pts_velocity;
    vector<int>           ids;
    vector<int>           track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr  m_camera;
    double                cur_time{};
    double                prev_time{};

    int                              n_id;
    cv::Ptr<cv::FastFeatureDetector> p_fast_feature_detector;

    std::shared_ptr<ThreadPool> pool;

    std::vector<cv::Rect> grids_rect;
    std::vector<int>      grids_track_num;
    std::vector<bool>     grids_texture_status;  // true: abundant texture grid; false:
                                                 // textureless grid
    int     grid_height{};
    int     grid_width{};
    int     grid_res_height{};
    int     grid_res_width{};
    int     grids_threshold{};
    cv::Mat grids_detector_img;
};
