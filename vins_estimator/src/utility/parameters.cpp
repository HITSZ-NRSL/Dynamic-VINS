#include "parameters.h"

#include <thread>

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double      BIAS_ACC_THRESHOLD;
double      BIAS_GYR_THRESHOLD;
double      SOLVER_TIME;
int         NUM_ITERATIONS;
int         ESTIMATE_EXTRINSIC;
int         ESTIMATE_TD;
int         ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMAGE_TOPIC;
std::string DEPTH_TOPIC;
std::string IMU_TOPIC;

int             MAX_CNT;
int             MAX_CNT_SET;
int             MIN_DIST;
int             FREQ;
double          F_THRESHOLD;
int             SHOW_TRACK;
int             EQUALIZE;
int             FISHEYE;
std::string     FISHEYE_MASK;
std::string     CAM_NAMES;
int             STEREO_TRACK;
bool            PUB_THIS_FRAME;
Eigen::Matrix3d Ric;

double ROW, COL;
int    IMAGE_SIZE;
double TD, TR;

double         DEPTH_MIN_DIST;
double         DEPTH_MAX_DIST;
unsigned short DEPTH_MIN_DIST_MM;
unsigned short DEPTH_MAX_DIST_MM;

std::vector<std::string> SEMANTIC_LABEL;
std::vector<std::string> STATIC_LABEL;
std::vector<std::string> DYNAMIC_LABEL;

int NUM_GRID_ROWS;
int NUM_GRID_COLS;

int FRONTEND_FREQ;
int USE_IMU;
int NUM_THREADS;
int STATIC_INIT;

int FIX_DEPTH;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

// https://blog.csdn.net/qq_16952303/article/details/80259660
void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings;
    fsSettings.open(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    NUM_THREADS = fsSettings["num_threads"];
    if (NUM_THREADS <= 1)
    {
        NUM_THREADS = std::thread::hardware_concurrency();
    }

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["depth_topic"] >> DEPTH_TOPIC;

    MAX_CNT = fsSettings["max_cnt"];

    MAX_CNT_SET = MAX_CNT;

    MIN_DIST = fsSettings["min_dist"];

    FREQ                         = fsSettings["freq"];
    F_THRESHOLD                  = fsSettings["F_threshold"];
    SHOW_TRACK                   = fsSettings["show_track"];
    EQUALIZE                     = fsSettings["equalize"];
    FISHEYE                      = fsSettings["fisheye"];
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES = config_file;

    DEPTH_MIN_DIST = fsSettings["depth_min_dist"];
    DEPTH_MAX_DIST = fsSettings["depth_max_dist"];

    DEPTH_MIN_DIST_MM = DEPTH_MIN_DIST * 1000;
    DEPTH_MAX_DIST_MM = DEPTH_MAX_DIST * 1000;

    NUM_GRID_ROWS = fsSettings["num_grid_rows"];
    NUM_GRID_COLS = fsSettings["num_grid_cols"];
    ROS_INFO("NUM_GRID_ROWS: %d, NUM_GRID_COLS: %d", NUM_GRID_ROWS, NUM_GRID_COLS);

    FRONTEND_FREQ = fsSettings["frontend_freq"];
    ROS_INFO("FRONTEND_FREQ: %d", FRONTEND_FREQ);

    STEREO_TRACK   = false;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    SOLVER_TIME    = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX   = fsSettings["keyframe_parallax"];
    ROS_INFO("keyframe_parallax: %f", MIN_PARALLAX);
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    USE_IMU = fsSettings["imu"];
    ROS_INFO("USE_IMU: %d\n", USE_IMU);
    if (USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    ROW        = fsSettings["image_height"];
    COL        = fsSettings["image_width"];
    IMAGE_SIZE = ROW * COL;
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    for (auto iter : fsSettings["semantic_label"])
    {
        SEMANTIC_LABEL.emplace_back(iter.string());
    }

    for (auto iter : fsSettings["static_label"])
    {
        STATIC_LABEL.emplace_back(iter.string());
    }

    for (auto iter : fsSettings["dynamic_label"])
    {
        DYNAMIC_LABEL.emplace_back(iter.string());
    }

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.emplace_back(Eigen::Matrix3d::Identity());
        TIC.emplace_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.txt";
    }
    else
    {
        if (ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.txt";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        Ric     = eigen_R;
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
    }

    INIT_DEPTH         = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD          = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }

    STATIC_INIT = fsSettings["static_init"];
    if (!fsSettings["fix_depth"].empty())
        FIX_DEPTH = fsSettings["fix_depth"];
    else
        FIX_DEPTH = 1;
    fsSettings.release();
}
