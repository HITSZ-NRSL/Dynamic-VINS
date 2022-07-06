#pragma once

#include <mutex>
#include <thread>

#include "../feature_manager/feature_manager.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../initial/initial_sfm.h"
#include "../initial/solve_5pts.h"
#include "../utility/parameters.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"

#include "../feature_tracker/feature_tracker.h"

#include <sensor_msgs/Imu.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>

#include "../factor/imu_factor.h"
#include "../factor/marginalization_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/projection_factor.h"
#include "../factor/projection_td_factor.h"
#include <ceres/ceres.h>

#include <opencv2/core/eigen.hpp>
#include <queue>
#include <unordered_map>

#include <sophus/se3.h>
#include <sophus/so3.h>

class Estimator
{
public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration,
                    const Vector3d &angular_velocity);

    void processImage(map<int, Eigen::Matrix<double, 7, 1>> &image, const std_msgs::Header &header);

    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points,
                      Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();

    bool initialStructure();

    bool visualInitialAlign();

    bool visualInitialAlignWithDepth();

    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    void slideWindow();

    void slideWindowNew();

    void slideWindowOld();

    void solveOdometry();

    void optimization();

    void vector2double();

    void double2vector();

    bool failureDetection();

    bool staticInitialAlignWithDepth();

    void updateLatestStates();

    Matrix3d predictMotion(double t0, double t1);

    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);

    void predict(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);

    bool IMUAvailable(double t);

    bool initialStructureWithDepth();
    void movingConsistencyCheck(set<int> &removeIndex);

    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                             Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                             double depth, Vector3d &uvi, Vector3d &uvj);
    double reprojectionError3D(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                               Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                               double depth, Vector3d &uvi, Vector3d &uvj);

    bool
    getIMUInterval(double t0, double t1,
                   std::vector<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> &imu_vector);
    void
    initFirstIMUPose(std::vector<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> &imu_vector);
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD        = 0,
        MARGIN_SECOND_NEW = 1
    };

    bool           openExEstimation;
    FeatureTracker featureTracker;

    SolverFlag          solver_flag;
    MarginalizationFlag marginalization_flag;
    Vector3d            g;
    // extrinsic
    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    // VIO state vector
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double   td{};

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double   Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]{};
    Vector3d         acc_0, gyr_0;

    vector<double>   dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count{};  // cl:滑动窗口中帧的数目,最大为滑窗大小
    int sum_of_back{}, sum_of_front{}, sum_of_invalid{};

    FeatureManager    f_manager;
    MotionEstimator   m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu{};
    bool is_valid{};
    bool failure_occur{};

    vector<Vector3d> key_poses;
    double           initial_timestamp{};

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE]{};

    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS]{};
    double para_Feature[NUM_OF_F][SIZE_FEATURE]{};
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE]{};
    double para_Td[1][1]{};
    int    find_solved[WINDOW_SIZE + 1]{};

    MarginalizationInfo *last_marginalization_info{};
    vector<double *>     last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase        *tmp_pre_integration{};

    // relocalization variable
    bool             relocalization_info{};
    double           relo_frame_stamp{};
    double           relo_frame_index{};
    int              relo_frame_local_index{};
    vector<Vector3d> match_points;
    double           relo_Pose[SIZE_POSE]{};
    Matrix3d         drift_correct_r;
    Vector3d         drift_correct_t;
    Vector3d         prev_relo_t;
    Matrix3d         prev_relo_r;
    Vector3d         relo_relative_t;
    Quaterniond      relo_relative_q;
    double           relo_relative_yaw{};

    std::mutex m_imu, m_propagate;

    bool   init_imu{};
    double prevTime = -1;

    queue<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> imu_buf;

    double             latest_time{};
    Eigen::Vector3d    latest_P;
    Eigen::Quaterniond latest_Q;
    Eigen::Vector3d    latest_V;
    Eigen::Vector3d    latest_Ba;
    Eigen::Vector3d    latest_Bg;
    bool               initFirstPoseFlag{};
};
