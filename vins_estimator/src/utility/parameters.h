#pragma once

#include "utility.h"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <vector>

const double FOCAL_LENGTH = 460.0;
const int    WINDOW_SIZE  = 10;
const int    NUM_OF_CAM   = 1;
const int    NUM_OF_F     = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int    ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d              G;

extern double      BIAS_ACC_THRESHOLD;
extern double      BIAS_GYR_THRESHOLD;
extern double      SOLVER_TIME;
extern int         NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string IMAGE_TOPIC;
extern std::string DEPTH_TOPIC;
extern std::string IMU_TOPIC;
extern double      TD;
extern double      TR;
extern int         ESTIMATE_TD;
extern int         ROLLING_SHUTTER;
extern double      ROW, COL;

extern int IMAGE_SIZE;

extern double          DEPTH_MIN_DIST;
extern double          DEPTH_MAX_DIST;
extern unsigned short  DEPTH_MIN_DIST_MM;
extern unsigned short  DEPTH_MAX_DIST_MM;
extern int             MAX_CNT;
extern int             MAX_CNT_SET;
extern int             MIN_DIST;
extern int             FREQ;
extern double          F_THRESHOLD;
extern int             SHOW_TRACK;
extern int             EQUALIZE;
extern int             FISHEYE;
extern std::string     FISHEYE_MASK;
extern std::string     CAM_NAMES;
extern int             STEREO_TRACK;
extern bool            PUB_THIS_FRAME;
extern Eigen::Matrix3d Ric;

extern std::vector<std::string> SEMANTIC_LABEL;
extern std::vector<std::string> STATIC_LABEL;
extern std::vector<std::string> DYNAMIC_LABEL;

extern int NUM_GRID_ROWS;
extern int NUM_GRID_COLS;

extern int FRONTEND_FREQ;

extern int USE_IMU;
extern int NUM_THREADS;

extern int STATIC_INIT;

extern int FIX_DEPTH;

void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE      = 7,
    SIZE_SPEEDBIAS = 9,
    //    SIZE_SPEED = 3,
    //    SIZE_BIAS = 6,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P  = 0,
    O_R  = 3,
    O_V  = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
