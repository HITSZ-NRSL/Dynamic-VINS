#include "estimator.h"
#include "../utility/visualization.h"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <utility>

Estimator::Estimator() : f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info   = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td                            = TD;
    g                             = G;

    featureTracker.readIntrinsicParameter(CAM_NAMES);
    if (FISHEYE)
    {
        featureTracker.fisheye_mask = cv::imread(FISHEYE_MASK, 0);
        if (!featureTracker.fisheye_mask.data)
        {
            ROS_INFO("load mask fail");
            ROS_BREAK();
        }
        else
            ROS_INFO("load mask success");
    }
    featureTracker.initGridsDetector();
}

void Estimator::clearState()
{
    m_imu.lock();
    while (!imu_buf.empty())
        imu_buf.pop();
    m_imu.unlock();

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;

        // cl
        find_solved[i] = 0;
        // cl
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }
    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }
    first_imu = false, sum_of_back = 0;
    sum_of_front      = 0;
    frame_count       = 0;
    solver_flag       = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    openExEstimation = false;

    delete tmp_pre_integration;

    delete last_marginalization_info;

    tmp_pre_integration       = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur       = false;
    relocalization_info = false;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();

    latest_Q = Eigen::Quaterniond(1, 0, 0, 0);

    init_imu = true;

    prevTime = -1;

    initFirstPoseFlag = false;
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration,
                           const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0     = linear_acceleration;
        gyr_0     = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] =
            new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        // if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int      j        = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr   = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc   = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(map<int, Eigen::Matrix<double, 7, 1>> &image,
                             const std_msgs::Header                &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    // FeaturePerFrame
    // FeaturePerId
    // feature
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header.stamp.toSec();

    if (USE_IMU)
    {
        double curTime = header.stamp.toSec() + td;

        while (!IMUAvailable(curTime))
        {
            printf("waiting for imu ... \r");
            std::chrono::milliseconds dura(2);
            std::this_thread::sleep_for(dura);
        }

        std::vector<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> imu_vector;
        getIMUInterval(prevTime, curTime, imu_vector);
        if (!initFirstPoseFlag)
            initFirstIMUPose(imu_vector);
        for (size_t i = 0; i < imu_vector.size(); i++)
        {
            double dt;
            if (i == 0)
                dt = imu_vector[i].first - prevTime;
            else if (i == imu_vector.size() - 1)
                dt = curTime - imu_vector[i - 1].first;
            else
                dt = imu_vector[i].first - imu_vector[i - 1].first;
            processIMU(dt, imu_vector[i].second.first, imu_vector[i].second.second);
        }
        prevTime = curTime;
    }

    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres =
                f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(
                    corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0]             = calib_ric;
                RIC[0]             = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    TicToc opt_time;
    if (solver_flag == INITIAL)
    {
        if (USE_IMU && !STATIC_INIT)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                if (ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
                {
                    result            = initialStructure();
                    initial_timestamp = header.stamp.toSec();
                }
                // if init sfm success
                if (result)
                {
                    solver_flag = NON_LINEAR;
                    solveOdometry();
                    slideWindow();
                    f_manager.removeFailures();
                    ROS_INFO("Initialization finish!");
                    last_R  = Rs[WINDOW_SIZE];
                    last_P  = Ps[WINDOW_SIZE];
                    last_R0 = Rs[0];
                    last_P0 = Ps[0];
                }
                else
                    slideWindow();
            }
            else
                frame_count++;
        }
        else
        {
            f_manager.triangulateWithDepth(Ps, tic, ric);

            optimization();
            if (USE_IMU)
            {
                if (frame_count == WINDOW_SIZE)
                {
                    int i = 0;
                    for (auto &frame_it : all_image_frame)
                    {
                        frame_it.second.R = Rs[i];
                        frame_it.second.T = Ps[i];
                        i++;
                    }
                    if (ESTIMATE_EXTRINSIC != 2)
                    {
                        solveGyroscopeBias(all_image_frame, Bgs);
                        for (int j = 0; j <= WINDOW_SIZE; j++)
                        {
                            pre_integrations[j]->repropagate(Vector3d::Zero(), Bgs[j]);
                        }
                        optimization();
                        solver_flag = NON_LINEAR;
                        slideWindow();
                        ROS_INFO("Initialization finish!");
                        last_R  = Rs[WINDOW_SIZE];
                        last_P  = Ps[WINDOW_SIZE];
                        last_R0 = Rs[0];
                        last_P0 = Ps[0];
                    }
                }
            }
            else
            {
                if (frame_count == WINDOW_SIZE)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
            }

            if (frame_count < WINDOW_SIZE)
            {
                frame_count++;
                int prev_frame   = frame_count - 1;
                Ps[frame_count]  = Ps[prev_frame];
                Vs[frame_count]  = Vs[prev_frame];
                Rs[frame_count]  = Rs[prev_frame];
                Bas[frame_count] = Bas[prev_frame];
                Bgs[frame_count] = Bgs[prev_frame];
            }
        }
    }
    else
    {
        TicToc t_solve;
        if (!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);

        f_manager.triangulateWithDepth(Ps, tic, ric);
        optimization();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        {
            set<int> removeIndex;
            movingConsistencyCheck(removeIndex);
            if (SHOW_TRACK)
            {
                for (auto iter = image.begin(), iter_next = image.begin(); iter != image.end();
                     iter = iter_next)
                {
                    ++iter_next;
                    auto it = removeIndex.find(iter->first);

                    if (it != removeIndex.end())
                    {
                        image.erase(iter);
                    }
                }
            }
        }

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = true;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }
        f_manager.removeFailures();

        // TicToc t_margin;
        slideWindow();
        // ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.emplace_back(Ps[i]);

        last_R  = Rs[WINDOW_SIZE];
        last_P  = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }

    static double whole_opt_time = 0;
    static size_t cnt_frame      = 0;
    ++cnt_frame;
    whole_opt_time += opt_time.toc();
    ROS_DEBUG("average opt costs: %f", whole_opt_time / cnt_frame);
}

/**
 * @brief   视觉的结构初始化
 * @Description 确保IMU有充分运动激励
 *              relativePose()找到具有足够视差的两帧,由F矩阵恢复R、t作为初始值
 *              sfm.construct() 全局纯视觉SFM 恢复滑动窗口帧的位姿
 *              visualInitialAlign()视觉惯性联合初始化
 * @return  bool true:初始化成功
 */
bool Estimator::initialStructure()
{
    // check imu observibility
    bool is_imu_excited = false;
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d                          sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end();
             frame_it++)
        {
            double   dt    = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g     = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end();
             frame_it++)
        {
            double   dt    = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));  // 标准差
        // ROS_WARN("IMU variation %f!", var);
        if (var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            // return false;
        }
        else
        {
            is_imu_excited = true;
        }
    }

    TicToc t_sfm;
    // global sfm
    Quaterniond        Q[frame_count + 1];
    Vector3d           T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int        imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id    = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.emplace_back(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()});
            tmp_feature.observation_depth.emplace_back(imu_j, it_per_frame.depth);
        }
        sfm_f.emplace_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int      l;
    //保证具有足够的视差,由F矩阵恢复Rt
    //第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用
    //此处的relative_R，relative_T为当前帧到参考帧（第l帧）的坐标系变换Rt
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        // ROS_INFO("Not enough features!");
        return false;
    }

    //对窗口中每个图像帧求解sfm问题
    //得到所有图像帧相对于参考帧的姿态四元数Q、平移向量T和特征点坐标sfm_tracked_points。
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    //对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化,得到每一帧的姿态
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator      it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R            = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T            = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        // points: map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            it             = sfm_tracked_points.find(feature_id);
            if (it != sfm_tracked_points.end())
            {
                Vector3d    world_pts = it->second;
                cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                pts_3_vector.push_back(pts_3);
                Vector2d    img_pts = id_pts.second.head<2>();
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        /**
         *bool cv::solvePnP(    求解pnp问题
         *   InputArray  objectPoints,   特征点的3D坐标数组
         *   InputArray  imagePoints,    特征点对应的图像坐标
         *   InputArray  cameraMatrix,   相机内参矩阵
         *   InputArray  distCoeffs,     失真系数的输入向量
         *   OutputArray     rvec,       旋转向量
         *   OutputArray     tvec,       平移向量
         *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
         *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
         *)
         */
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        //这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp              = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // Rs Ps ric init
    //进行视觉惯性联合初始化
    if (visualInitialAlignWithDepth())
    {
        if (!is_imu_excited)
        {
            // 利用加速度平均值估计Bas
            Vector3d sum_a(0, 0, 0);
            for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end();
                 frame_it++)
            {
                double   dt    = frame_it->second.pre_integration->sum_dt;
                Vector3d tmp_a = frame_it->second.pre_integration->delta_v / dt;
                sum_a += tmp_a;
            }
            Vector3d avg_a;
            avg_a = sum_a * 1.0 / ((int)all_image_frame.size() - 1);

            Vector3d tmp_Bas = avg_a - Utility::g2R(avg_a).inverse() * G;
            ROS_WARN_STREAM("accelerator bias initial calibration " << tmp_Bas.transpose());
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                Bas[i] = tmp_Bas;
            }
        }
        return true;
    }
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

bool Estimator::initialStructureWithDepth()
{
    // check imu observibility
    bool is_imu_excited = false;

    map<double, ImageFrame>::iterator frame_it;
    Vector3d                          sum_a;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end();
         frame_it++)
    {
        double   dt    = frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
        sum_a += tmp_g;
    }
    Vector3d aver_g;
    aver_g     = sum_a * 1.0 / ((int)all_image_frame.size() - 1);
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end();
         frame_it++)
    {
        double   dt    = frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
        var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
    }
    var = sqrt(var / ((int)all_image_frame.size() - 1));  // 标准差
    // ROS_WARN("IMU variation %f!", var);
    if (var < 0.25)
    {
        ROS_INFO("IMU excitation not enouth!");
        // return false;
    }
    else
    {
        is_imu_excited = true;
    }
    TicToc t_sfm;
    // global sfm
    Quaterniond        Q[frame_count + 1];
    Vector3d           T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int        imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id    = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.emplace_back(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()});
            tmp_feature.observation_depth.emplace_back(imu_j, it_per_frame.depth);
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int      l;
    //保证具有足够的视差,由F矩阵恢复Rt
    //第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用
    //此处的relative_R，relative_T为当前帧到参考帧（第l帧）的坐标系变换Rt
    if (!relativePose(relative_R, relative_T, l))
    {
        //        ROS_INFO("Not enough features or parallax; Move device
        //        around");
        ROS_INFO("Not enough features!");
        return false;
    }

    //对窗口中每个图像帧求解sfm问题
    //得到所有图像帧相对于参考帧的姿态四元数Q、平移向量T和特征点坐标sfm_tracked_points。
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    //对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化,得到每一帧的姿态
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R            = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T            = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        // points: map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            it             = sfm_tracked_points.find(feature_id);
            if (it != sfm_tracked_points.end())
            {
                Vector3d    world_pts = it->second;
                cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                pts_3_vector.push_back(pts_3);
                //                Vector2d img_pts = id_pts.second.head<2>();
                //                cout << endl << "id_pts.second.head<2>():" <<
                //                id_pts.second.head<2>() << endl; cout << endl
                //                << "id_pts.second.head<0>():" <<
                //                id_pts.second.head<0>()
                //                << endl;
                cv::Point2f pts_2(id_pts.second(3), id_pts.second(3));
                pts_2_vector.push_back(pts_2);
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        /**
         *bool cv::solvePnP(    求解pnp问题
         *   InputArray  objectPoints,   特征点的3D坐标数组
         *   InputArray  imagePoints,    特征点对应的图像坐标
         *   InputArray  cameraMatrix,   相机内参矩阵
         *   InputArray  distCoeffs,     失真系数的输入向量
         *   OutputArray     rvec,       旋转向量
         *   OutputArray     tvec,       平移向量
         *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
         *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
         *)
         */
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        //这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp              = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    // if (is_imu_excited) {
    //   if (visualInitialAlignWithDepth())
    //     return true;
    // } else {
    //   staticInitialAlignWithDepth();
    //   return true;
    // }
    // ROS_INFO("misalign visual structure with IMU");
    // return false;

    if (visualInitialAlignWithDepth())
    {
        if (!is_imu_excited)
        {
            Vector3d tmp_Bas = aver_g - Utility::g2R(aver_g).inverse() * G;
            ROS_WARN_STREAM("accelerator bias initial calibration " << tmp_Bas.transpose());
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                Bas[i] = tmp_Bas;
            }
        }
        return true;
    }
    else
    {
        staticInitialAlignWithDepth();
        return true;
    }

    ROS_INFO("misalign visual structure with IMU");
    return false;
}

bool Estimator::visualInitialAlign()
{
    TicToc   t_g;
    VectorXd x;
    // solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        ROS_ERROR("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri                              = all_image_frame[Headers[i]].R;
        Vector3d Pi                              = all_image_frame[Headers[i]].T;
        Ps[i]                                    = Pi;
        Rs[i]                                    = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    // triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    // f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));
    f_manager.triangulateWithDepth(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    // ROS_DEBUG("the scale is %f\n", s);
    //  do repropagate here
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int                               kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // if (it_per_id.used_num < 4)
        //     continue;
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Matrix3d R0  = Utility::g2R(g);
    double   yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0           = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g            = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

/**
 * @brief   视觉惯性联合初始化
 * @Description 陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *              更新了Bgs后，IMU测量量需要repropagate
 *              得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return  bool true：成功
 */
bool Estimator::staticInitialAlignWithDepth()
{
    // 利用加速度平均值估计Bgs, Bas, g
    map<double, ImageFrame>::iterator frame_it;
    Vector3d                          sum_a(0, 0, 0);
    Vector3d                          sum_w(0, 0, 0);
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end();
         frame_it++)
    {
        sum_a +=
            frame_it->second.pre_integration->delta_v / frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_w;
        for (auto &gyr_msg : frame_it->second.pre_integration->gyr_buf)
        {
            tmp_w += gyr_msg;
        }
        sum_w += tmp_w / frame_it->second.pre_integration->gyr_buf.size();
    }
    Vector3d avg_a   = sum_a * 1.0 / ((int)all_image_frame.size() - 1);
    Vector3d avg_w   = sum_w * 1.0 / ((int)all_image_frame.size() - 1);
    g                = avg_a.normalized() * G.z();
    Vector3d tmp_Bas = avg_a - g;

    // solveGyroscopeBias(all_image_frame, Bgs);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << avg_w.transpose());
    ROS_WARN_STREAM("accelerator bias initial calibration " << tmp_Bas.transpose());
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Bgs[i] = avg_w;
        Bas[i] = tmp_Bas;
    }

    //陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Bas[i], Bgs[i]);
    }

    //通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0  = Utility::g2R(g);
    double   yaw = Utility::R2ypr(R0).x();
    R0           = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g            = R0 * G;

    Matrix3d rot_diff = R0;
    //所有变量从参考坐标系c0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        ROS_ERROR("%d farme's Ps is %f | %f | %f\n", i, Ps[i].x(), Ps[i].y(),
                  Ps[i].z());  // shan add
        ROS_ERROR("%d farme's Vs is %f | %f | %f\n", i, Vs[i].x(), Vs[i].y(), Vs[i].z());
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
        // Vs[i] = Vector3d(0, 0, 0);
    }
    ROS_DEBUG_STREAM("static g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

/**
 * @brief   视觉惯性联合初始化
 * @Description 陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *              更新了Bgs后，IMU测量量需要repropagate
 *              得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return  bool true：成功
 */
bool Estimator::visualInitialAlignWithDepth()
{
    TicToc   t_g;
    VectorXd x;

    // solve scale
    //计算陀螺仪偏置，尺度，重力加速度和速度
    solveGyroscopeBias(all_image_frame, Bgs);

    if (!LinearAlignmentWithDepth(all_image_frame, g, x))
    {
        ROS_ERROR("solve g failed!");
        return false;
    }

    // do repropagate here
    //陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // change state
    // 得到所有图像帧的位姿Ps、Rs，并将其置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri                              = all_image_frame[Headers[i]].R;
        Vector3d Pi                              = all_image_frame[Headers[i]].T;
        Ps[i]                                    = Pi;
        Rs[i]                                    = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    //将所有特征点的深度置为-1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    // triangulat on cam pose , no tic
    //重新计算特征点的深度
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (auto &i : TIC_TMP)
        i.setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    // f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));
    f_manager.triangulateWithDepth(Ps, &(TIC_TMP[0]), &(RIC[0]));

    // do repropagate here
    //陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // ROS_ERROR("before %f | %f | %f\n", Ps[1].x(), Ps[1].y(), Ps[1].z());//shan
    // add 将Ps、Vs、depth尺度s缩放
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = Ps[i] - Rs[i] * TIC[0] - (Ps[0] - Rs[0] * TIC[0]);
    // ROS_ERROR("after  %f | %f | %f\n", Ps[1].x(), Ps[1].y(), Ps[1].z());//shan
    // add
    int                               kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    //通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0  = Utility::g2R(g);
    double   yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0           = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g            = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    //所有变量从参考坐标系c0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
        // ROS_ERROR("%d farme's Ps is %f | %f | %f\n", i, Ps[i].x(), Ps[i].y(),
        //           Ps[i].z()); // shan add
        // ROS_ERROR("%d farme's Vs is %f | %f | %f\n", i, Vs[i].x(), Vs[i].y(),
        //           Vs[i].z());
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

/**
 * @brief   判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
 * @Description    判断每帧到窗口最后一帧对应特征点的平均视差是否大于30
                solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
 * @param[out]   relative_R 当前帧到第l帧之间的旋转矩阵R
 * @param[out]   relative_T 当前帧到第l帧之间的平移向量T
 * @param[out]   L 保存滑动窗口中与当前帧满足初始化条件的那一帧
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with
    // newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        // corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        corres = f_manager.getCorrespondingWithDepth(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (auto &corre : corres)
            {
                Vector2d pts_0(corre.first(0) / corre.first(2), corre.first(1) / corre.first(2));
                Vector2d pts_1(corre.second(0) / corre.second(2),
                               corre.second(1) / corre.second(2));
                // Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                // Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax    = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 &&
                m_estimator.solveRelativeRT_PNP(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to "
                          "triangulate the whole structure",
                          average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulateWithDepth(Ps, tic, ric);
        //        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if (USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

// 数据转换，vector2double的相反过程
// 同时这里为防止优化结果往零空间变化，会根据优化前后第一帧的位姿差进行修正。
void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0     = Utility::R2ypr(last_R0);
        origin_P0     = last_P0;
        failure_occur = false;
    }
    if (USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(
            Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5])
                .toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        // TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4],
                                           para_Pose[0][5])
                                   .toRotationMatrix()
                                   .transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = rot_diff *
                    Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5])
                        .normalized()
                        .toRotationMatrix();

            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                        para_Pose[i][1] - para_Pose[0][1],
                                        para_Pose[i][2] - para_Pose[0][2]) +
                    origin_P0;

            Vs[i] = rot_diff *
                    Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8]);
        }

        // relative info between two loop frame
        if (relocalization_info)
        {
            Matrix3d relo_r;
            Vector3d relo_t;
            relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5])
                                    .normalized()
                                    .toRotationMatrix();
            relo_t =
                rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0], relo_Pose[1] - para_Pose[0][1],
                                    relo_Pose[2] - para_Pose[0][2]) +
                origin_P0;
            double drift_correct_yaw;
            drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
            drift_correct_r   = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
            drift_correct_t   = prev_relo_t - drift_correct_r * relo_t;
            relo_relative_t   = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
            relo_relative_q   = relo_r.transpose() * Rs[relo_frame_local_index];
            relo_relative_yaw = Utility::normalizeAngle(
                Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
            // cout << "vins relo " << endl;
            // cout << "vins relative_t " << relo_relative_t.transpose() << endl;
            // cout << "vins relative_yaw " <<relo_relative_yaw << endl;
            relocalization_info = 0;
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5])
                        .normalized()
                        .toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }

        // relative info between two loop frame
        if (relocalization_info)
        {
            Matrix3d relo_r;
            Vector3d relo_t;
            relo_r = Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5])
                         .normalized()
                         .toRotationMatrix();
            relo_t = Vector3d(relo_Pose[0], relo_Pose[1], relo_Pose[2]);
            double drift_correct_yaw;
            drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
            drift_correct_r   = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
            drift_correct_t   = prev_relo_t - drift_correct_r * relo_t;
            relo_relative_t   = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
            relo_relative_q   = relo_r.transpose() * Rs[relo_frame_local_index];
            relo_relative_yaw = Utility::normalizeAngle(
                Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
            // cout << "vins relo " << endl;
            // cout << "vins relative_t " << relo_relative_t.transpose() << endl;
            // cout << "vins relative_yaw " <<relo_relative_yaw << endl;
            relocalization_info = false;
        }
    }
    if (USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0], para_Ex_Pose[i][1], para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6], para_Ex_Pose[i][3], para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5])
                         .normalized()
                         .toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD && USE_IMU)
        td = para_Td[0][0];
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        // return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d    tmp_R   = Rs[WINDOW_SIZE];
    Matrix3d    delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double      delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        // return true;
    }
    return false;
}

void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem       problem;
    ceres::LossFunction *loss_function;
    // loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    /*######优化参数：q、p；v、Ba、Bg#######*/
    //添加ceres参数块
    //因为ceres用的是double数组，所以在下面用vector2double做类型装换
    // Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if (USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i],
                                      SIZE_SPEEDBIAS);  // v、Ba、Bg参数
    }
    if (!USE_IMU)
    {
        problem.SetParameterBlockConstant(para_Pose[0]);
    }
    /*######优化参数：imu与camera外参#######*/
    for (auto &i : para_Ex_Pose)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(i, SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) ||
            openExEstimation)
        {
            // ROS_DEBUG("estimate extinsic param");
            openExEstimation = true;
        }
        else
        {
            // ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(i);
        }
    }
    /*######优化参数：imu与camera之间的time offset#######*/
    if (USE_IMU)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //速度过低时，不估计td
        if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        {
            problem.SetParameterBlockConstant(para_Td[0]);
        }
    }

    //构建残差
    /*******先验残差*******/
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor =
            new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    /*******预积分残差*******/
    if (USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)  //预积分残差，总数目为frame_count
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt >
                10.0)  //两图像帧之间时间过长，不使用中间的预积分 tzhang
                continue;
            IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
            //添加残差格式：残差因子，鲁棒核函数，优化变量（i时刻位姿，i时刻速度与偏置，i+1时刻位姿，i+1时刻速度与偏置）
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i],
                                     para_Pose[j], para_SpeedBias[j]);
        }
    }

    /*******重投影残差*******/
    //重投影残差相关，此时使用了Huber损失核函数
    int f_m_cnt       = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        if (it_per_id.is_dynamic)
        {
            continue;
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        // if (it_per_id.used_num < 4)
        //     continue;
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)  //遍历观测到路标点的图像帧
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;  //测量值
            if (ESTIMATE_TD)
            {
                ProjectionTdFactor *f_td = new ProjectionTdFactor(
                    pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                    it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j],
                                         para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                if (it_per_id.estimate_flag == 1 && FIX_DEPTH)
                    problem.SetParameterBlockConstant(para_Feature[feature_index]);
                else if (it_per_id.estimate_flag == 2)
                {
                    problem.SetParameterUpperBound(para_Feature[feature_index], 0,
                                                   2 / DEPTH_MAX_DIST);
                }
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j],
                                         para_Ex_Pose[0], para_Feature[feature_index]);
                if (it_per_id.estimate_flag == 1 && FIX_DEPTH)
                    problem.SetParameterBlockConstant(para_Feature[feature_index]);
                else if (it_per_id.estimate_flag == 2)
                {
                    // 防止远处的点深度过小
                    problem.SetParameterUpperBound(para_Feature[feature_index], 0,
                                                   2 / DEPTH_MAX_DIST);
                }
            }
            f_m_cnt++;
        }
    }
    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    //添加闭环检测残差，计算滑动窗口中与每一个闭环关键帧的相对位姿，这个相对位置是为后面的图优化准备
    if (relocalization_info)
    {
        // printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index         = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            if (it_per_id.is_dynamic)
            {
                continue;
            }
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            // if (it_per_id.used_num < 4)
            //     continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if (start <= relo_frame_local_index)
            {
                while ((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if ((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(),
                                              match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose,
                                             para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }
            }
        }
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //    options.num_threads = 4;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations         = NUM_ITERATIONS;
    //    options.use_explicit_schur_complement = true;
    //    options.minimizer_progress_to_stdout = false;
    // options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc                 t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));

    // 防止优化结果在零空间变化，通过固定第一帧的位姿
    double2vector();

    if (frame_count < WINDOW_SIZE)
        return;

    TicToc t_whole_marginalization;
    //边缘化处理
    //如果次新帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验：
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        // 先验部分，基于先验残差，边缘化滑窗中第0帧时刻的状态向量
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor =
                new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if (USE_IMU)
        {
            // imu
            // 预积分部分，基于第0帧与第1帧之间的预积分残差，边缘化第0帧状态向量
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor         *imu_factor          = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                    imu_factor, NULL,
                    vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1],
                                     para_SpeedBias[1]},
                    vector<int>{0, 1});  //边缘化 para_Pose[0], para_SpeedBias[0]
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        //图像部分，基于与第0帧相关的图像残差，边缘化第一次观测的图像帧为第0帧的路标点和第0帧
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                if (it_per_id.is_dynamic)
                    continue;
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                // if (it_per_id.used_num < 4)
                //     continue;
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)  //仅处理第一次观测的图像帧为第0帧的情形
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame :
                     it_per_id.feature_per_frame)  //对观测到路标点的图像帧的遍历
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(
                            pts_i, pts_j, it_per_id.feature_per_frame[0].velocity,
                            it_per_frame.velocity, it_per_id.feature_per_frame[0].cur_td,
                            it_per_frame.cur_td, it_per_id.feature_per_frame[0].uv.y(),
                            it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                            f_td, loss_function,
                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0],
                                             para_Feature[feature_index], para_Td[0]},
                            vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor  *f                   = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                            f, loss_function,
                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0],
                                             para_Feature[feature_index]},
                            vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        //仅仅改变滑窗double部分地址映射，具体值的通过slideWindow和vector2double函数完成；记住边缘化仅仅改变A和b，不改变状态向量
        //由于第0帧观测到的路标点全被边缘化，即边缘化后保存的状态向量中没有路标点;因此addr_shift无需添加路标点
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)  //最老图像帧数据丢弃，从i=1开始遍历
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] =
                para_Pose[i - 1];  // i数据保存到1-1指向的地址，滑窗向前移动一格
            if (USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (auto &i : para_Ex_Pose)
            addr_shift[reinterpret_cast<long>(i)] = i;
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        delete last_marginalization_info;
        last_marginalization_info             = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else  //将次新的图像帧数据边缘化； tzhang
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks),
                       std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int>
                    drop_set;  //记录需要丢弃的变量在last_marginalization_parameter_blocks中的索引
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size());
                     i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] !=
                               para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor =
                    new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                    marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            //            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            //            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)  // WINDOW_SIZE - 1会被边缘化，不保存
                    continue;
                else if (i == WINDOW_SIZE)  // WINDOW_SIZE数据保存到WINDOW_SIZE-1指向的地址
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];

                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];

                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (auto &i : para_Ex_Pose)
                addr_shift[reinterpret_cast<long>(i)] = i;
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }

            vector<double *> parameter_blocks =
                marginalization_info->getParameterBlocks(addr_shift);

            delete last_marginalization_info;
            last_marginalization_info             = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());

    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)  // 边缘化最老的图像帧，即次新的图像帧为关键帧
    {
        double t_0 = Headers[0];
        back_R0    = Rs[0];
        back_P0    = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            // 1、滑窗中的数据往前移动一帧；运行结果就是WINDOW_SIZE位置的状态为之前0位置对应的状态
            //  0,1,2...WINDOW_SIZE——>1,2...WINDOW_SIZE,0
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Rs[i].swap(Rs[i + 1]);
                if (USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
                ++find_solved[i + 1];
                find_solved[i] = find_solved[i + 1];
            }
            // 2、处理前，WINDOW_SIZE位置的状态为之前0位置对应的状态；处理后，WINDOW_SIZE位置的状态为之前WINDOW_SIZE位置对应的状态;之前0位置对应的状态被剔除
            //  0,1,2...WINDOW_SIZE——>1,2...WINDOW_SIZE,WINDOW_SIZE
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE]      = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE]      = Rs[WINDOW_SIZE - 1];
            if (USE_IMU)
            {
                Vs[WINDOW_SIZE]  = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] =
                    new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            find_solved[WINDOW_SIZE] = 0;

            // 3、对时刻t_0(对应滑窗第0帧)之前的所有数据进行剔除；即all_image_frame中仅保留滑窗中图像帧0与图像帧WINDOW_SIZE之间的数据
            map<double, ImageFrame>::iterator it_0;
            it_0 = all_image_frame.find(t_0);
            delete it_0->second.pre_integration;
            it_0->second.pre_integration = nullptr;
            for (auto it = all_image_frame.begin(); it != it_0; ++it)
            {
                delete it->second.pre_integration;
                it->second.pre_integration = NULL;
            }
            all_image_frame.erase(all_image_frame.begin(), it_0);
            all_image_frame.erase(t_0);
            slideWindowOld();
        }
    }
    else  //边缘化次新的图像帧，主要完成的工作是数据衔接 tzhang
    {     // 0,1,2...WINDOW_SIZE-2, WINDOW_SIZE-1,
        // WINDOW_SIZE——>0,,1,2...WINDOW_SIZE-2,WINDOW_SIZE, WINDOW_SIZE
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1]      = Ps[frame_count];
            Rs[frame_count - 1]      = Rs[frame_count];

            find_solved[WINDOW_SIZE] = 0;

            if (USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double   tmp_dt                  = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity    = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration,
                                                                 tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }
                Vs[frame_count - 1]  = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] =
                    new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

/**
 * @brief   进行重定位
 * @optional
 * @param[in]   _frame_stamp    重定位帧时间戳
 * @param[in]   _frame_index    重定位帧索引值
 * @param[in]   _match_points   重定位帧的所有匹配点
 * @param[in]   _relo_t     重定位帧平移向量
 * @param[in]   _relo_r     重定位帧旋转矩阵
 * @return      void
 */
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points,
                             Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t  = std::move(_relo_t);
    prev_relo_r  = std::move(_relo_r);
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        if (relo_frame_stamp == Headers[i])
        {
            relo_frame_local_index = i;
            relocalization_info    = true;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration,
                         const Vector3d &angularVelocity)
{
    m_imu.lock();
    imu_buf.push(make_pair(t, make_pair(linearAcceleration, angularVelocity)));
    // imu_predict_buf.push(
    //     make_pair(t, make_pair(linearAcceleration, angularVelocity)));
    m_imu.unlock();

    if (solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        // predict imu (no residual error)
        m_propagate.lock();
        predict(t, linearAcceleration, angularVelocity);
        m_propagate.unlock();
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
    }
}

void Estimator::updateLatestStates()
{
    m_propagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P    = Ps[frame_count];
    latest_Q    = Rs[frame_count];
    latest_V    = Vs[frame_count];
    latest_Ba   = Bas[frame_count];
    latest_Bg   = Bgs[frame_count];

    if (USE_IMU)
    {
        m_imu.lock();
        queue<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> tmp_imu_buf = imu_buf;
        for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
            predict(tmp_imu_buf.front().first, imu_buf.front().second.first,
                    imu_buf.front().second.second);
        m_imu.unlock();
    }
    m_propagate.unlock();
}

Matrix3d Estimator::predictMotion(double t0, double t1)
{
    Matrix3d relative_R = Matrix3d::Identity();
    if (imu_buf.empty())
        return relative_R;

    bool            first_imu = true;
    double          prev_imu_time;
    Eigen::Vector3d prev_gyr;

    m_imu.lock();
    queue<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> imu_predict_buf = imu_buf;
    m_imu.unlock();
    if (t1 <= imu_predict_buf.back().first)
    {
        while (imu_predict_buf.front().first <= t0)
        {
            imu_predict_buf.pop();
        }
        while (imu_predict_buf.front().first <= t1 && !imu_predict_buf.empty())
        {
            double t = imu_predict_buf.front().first;

            Eigen::Vector3d angular_velocity = imu_predict_buf.front().second.second;

            // Eigen::Vector3d linear_acceleration{
            //     imu_predict_buf.front()->linear_acceleration.x,
            //     imu_predict_buf.front()->linear_acceleration.y,
            //     imu_predict_buf.front()->linear_acceleration.z};

            imu_predict_buf.pop();

            if (first_imu)
            {
                prev_imu_time = t;
                first_imu     = false;
                prev_gyr      = angular_velocity;
                continue;
            }
            double dt     = t - prev_imu_time;
            prev_imu_time = t;

            // Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

            Eigen::Vector3d un_gyr = 0.5 * (prev_gyr + angular_velocity) - latest_Bg;
            // tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

            // Eigen::Vector3d un_acc_1 =
            //     tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

            // un_acc = 0.5 * (un_acc_0 + un_acc_1);

            // tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
            // tmp_V = tmp_V + dt * un_acc;

            // acc_0 = linear_acceleration;
            prev_gyr = angular_velocity;

            // cl
            // Transform the mean angular velocity from the IMU
            // frame to the cam0 frames.
            // Compute the relative rotation.
            Vector3d cam0_angle_axisd = RIC.back().transpose() * un_gyr * dt;
            relative_R *= AngleAxisd(cam0_angle_axisd.norm(), cam0_angle_axisd.normalized())
                              .toRotationMatrix()
                              .transpose();
        }
    }

    return relative_R;
}

void Estimator::predict(double t, const Vector3d &linearAcceleration,
                        const Vector3d &angularVelocity)
{
    if (init_imu)
    {
        latest_time = t;
        init_imu    = false;
        return;
    }
    double dt                = t - latest_time;
    latest_time              = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr   = 0.5 * (gyr_0 + angularVelocity) - latest_Bg;
    latest_Q                 = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linearAcceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc   = 0.5 * (un_acc_0 + un_acc_1);
    latest_P                 = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V                 = latest_V + dt * un_acc;
}

bool Estimator::IMUAvailable(double t)
{
    if (!imu_buf.empty() && t <= imu_buf.back().first)
        return true;
    else
        return false;
}

void Estimator::initFirstIMUPose(
    std::vector<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> &imu_vector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    // return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int             n = (int)imu_vector.size();
    for (auto &i : imu_vector)
    {
        averAcc = averAcc + i.second.first;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0  = Utility::g2R(averAcc);
    double   yaw = Utility::R2ypr(R0).x();
    R0           = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0]        = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
}
bool Estimator::getIMUInterval(
    double t0, double t1,
    std::vector<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> &imu_vector)
{
    m_imu.lock();
    if (imu_buf.empty())
    {
        printf("not receive imu\n");
        m_imu.unlock();
        return false;
    }
    if (t1 <= imu_buf.back().first)
    {
        while (imu_buf.front().first <= t0)
        {
            imu_buf.pop();
        }
        while (imu_buf.front().first < t1)
        {
            imu_vector.emplace_back(std::move(imu_buf.front()));
            imu_buf.pop();
        }
        imu_vector.emplace_back(imu_buf.front());
        m_imu.unlock();
        return true;
    }
    else
    {
        printf("wait for imu\n");
        m_imu.unlock();
        return false;
    }
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                    Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w    = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj   = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double   rx       = residual.x();
    double   ry       = residual.y();
    return sqrt(rx * rx + ry * ry);
}

double Estimator::reprojectionError3D(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                      Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                      double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w  = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    return (pts_cj - uvj).norm();
}

void Estimator::movingConsistencyCheck(set<int> &removeIndex)
{
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        double depth = it_per_id.estimated_depth;

        double   err    = 0;
        double   err3D  = 0;
        int      errCnt = 0;
        int      imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                err += reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j],
                                         ric[0], tic[0], depth, pts_i, pts_j);

                err3D += reprojectionError3D(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j],
                                             Ps[imu_j], ric[0], tic[0], depth, pts_i, pts_j);
                errCnt++;
            }
        }
        if (errCnt > 0)
        {
            if (FOCAL_LENGTH * err / errCnt > 10 || err3D / errCnt > 5)
            {
                removeIndex.insert(it_per_id.feature_id);
                it_per_id.is_dynamic = true;
            }
            else
            {
                it_per_id.is_dynamic = false;
            }
        }
    }
}
