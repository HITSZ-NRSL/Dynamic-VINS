#include "feature_manager.h"
#include <complex>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <string>

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[]) : Rs(_Rs)
{
    for (auto &i : ric)
        i.setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }

    semantic_mask = cv::Mat(ROW, COL, CV_16UC1, cv::Scalar(0));
    semantic_id   = cv::Mat(ROW, COL, CV_16UC1, cv::Scalar(0));
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        if (it.is_dynamic)
        {
            continue;
        }
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

void FeatureManager::inputDepth(const cv::Mat &_depth_img)
{
    depth_img = _depth_img;
}

void FeatureManager::seedFilling(int x, int y, unsigned short last_depth, unsigned short _object_id)
{
    if (semantic_mask.at<unsigned short>(y, x) == 0)
    {
        unsigned short temp_depth = depth_img.at<unsigned short>(y, x);
        if (abs(temp_depth - last_depth) < 50 ||
            // temp_depth == 0 ||
            temp_depth > 3 * DEPTH_MAX_DIST_MM)
        {
            semantic_mask.at<unsigned short>(y, x) = _object_id;
            seedFilling(x - 1, y, temp_depth, _object_id);
            seedFilling(x + 1, y, temp_depth, _object_id);
            seedFilling(x, y - 1, temp_depth, _object_id);
            seedFilling(x, y + 1, temp_depth, _object_id);
        }
    }
}

void FeatureManager::setSemanticMask(int x1, int y1, int x2, int y2, bool is_compensate)
{
    if (!has_semantic)
    {
        semantic_mask    = cv::Mat(ROW, COL, CV_16UC1, cv::Scalar(0));
        prev_semantic_id = semantic_id;
        semantic_id      = cv::Mat(ROW, COL, CV_16UC1, cv::Scalar(0));
        has_semantic     = true;
    }
    unsigned short depth_corner = depth_img.at<unsigned short>(y1, x1);
    if (depth_img.at<unsigned short>(y1, x2) > depth_corner)
        depth_corner = depth_img.at<unsigned short>(y1, x2);
    if (depth_img.at<unsigned short>(y2, x1) > depth_corner)
        depth_corner = depth_img.at<unsigned short>(y2, x1);
    if (depth_img.at<unsigned short>(y2, x2) > depth_corner)
        depth_corner = depth_img.at<unsigned short>(y2, x2);

    int            x_center     = (x1 + x2) / 2;
    int            y_center     = (y1 + y2) / 2;
    unsigned short depth_center = depth_img.at<unsigned short>(y_center, x_center);

    if (depth_center > 0 && depth_corner > 0)
    {
        unsigned short depth_weight;
        int            delta = depth_corner - depth_center;
        if (delta > 1000)
        {
            depth_weight = (depth_corner + depth_center) / 2;
        }
        else
        {
            depth_weight = depth_center + 1000;
        }

        cv::rectangle(semantic_mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(depth_weight),
                      -1);
    }
    else if (depth_corner > 0)
    {
        cv::rectangle(semantic_mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(depth_corner),
                      -1);
    }
    else
    {
        cv::rectangle(semantic_mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0xffff), -1);
    }

    if (!is_compensate)
    {
        unsigned short temp_object_id = prev_semantic_id.at<unsigned short>(y_center, x_center);

        if (temp_object_id > 0)
        {
            dynamic_objects[temp_object_id].x_vel =
                x_center - dynamic_objects[temp_object_id].x_center;
            dynamic_objects[temp_object_id].y_vel =
                y_center - dynamic_objects[temp_object_id].y_center;
        }
        else
        {
            temp_object_id = ++object_id;
        }

        dynamic_objects[temp_object_id].x_center = x_center;
        dynamic_objects[temp_object_id].y_center = y_center;
        dynamic_objects[temp_object_id].x1       = x1;
        dynamic_objects[temp_object_id].y1       = y1;
        dynamic_objects[temp_object_id].x2       = x2;
        dynamic_objects[temp_object_id].y2       = y2;

        dynamic_objects[temp_object_id].x_weight_vel =
            (dynamic_objects[temp_object_id].x_weight_vel + dynamic_objects[temp_object_id].x_vel) /
            2;
        dynamic_objects[temp_object_id].y_weight_vel =
            (dynamic_objects[temp_object_id].y_weight_vel + dynamic_objects[temp_object_id].y_vel) /
            2;
        dynamic_objects[temp_object_id].is_update       = true;
        dynamic_objects[temp_object_id].no_update_times = 0;
        cv::rectangle(semantic_id, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(temp_object_id),
                      -1);
    }
}

void FeatureManager::compensateMissedDetection()
{
    if (dynamic_objects.empty())
    {
        return;
    }
    if (!has_semantic)
    {
        semantic_mask = cv::Mat(ROW, COL, CV_16UC1, cv::Scalar(0));
        has_semantic  = true;
    }
    for (auto iter = dynamic_objects.begin(), iter_next = dynamic_objects.begin();
         iter != dynamic_objects.end(); iter = iter_next)
    {
        ++iter_next;
        if (iter->second.is_update)
        {
            iter->second.is_update = false;
            continue;
        }

        if (iter->second.no_update_times > 2)
        {
            dynamic_objects.erase(iter);
            continue;
        }

        iter->second.x1 = (iter->second.x1 + iter->second.x_weight_vel - 10 > 0) ?
                              iter->second.x1 + iter->second.x_weight_vel - 10 :
                              0;
        iter->second.y1 = (iter->second.y1 + iter->second.y_weight_vel - 10 > 0) ?
                              iter->second.y1 + iter->second.y_weight_vel - 10 :
                              0;
        iter->second.x2 = (iter->second.x2 + iter->second.x_weight_vel + 10 < COL) ?
                              iter->second.x2 + iter->second.x_weight_vel + 10 :
                              COL - 1;
        iter->second.y2 = (iter->second.y2 + iter->second.y_weight_vel + 10 < ROW) ?
                              iter->second.y2 + iter->second.y_weight_vel + 10 :
                              ROW - 1;
        if (iter->second.x2 >= 0 && iter->second.x1 < COL && iter->second.y2 >= 0 &&
            iter->second.y1 < ROW)
            setSemanticMask(iter->second.x1, iter->second.y1, iter->second.x2, iter->second.y2,
                            true);

        iter->second.no_update_times++;
    }
}

bool FeatureManager::addFeatureCheckParallax(int                                    frame_count,
                                             map<int, Eigen::Matrix<double, 7, 1>> &image,
                                             double                                 td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int    parallax_num = 0;
    last_track_num      = 0;

    for (auto iter = image.begin(), iter_next = image.begin(); iter != image.end();
         iter = iter_next)
    {
        ++iter_next;

        unsigned short pt_depth_mm =
            depth_img.at<unsigned short>((int)iter->second(4), (int)iter->second(3));

        double pt_depth_m = pt_depth_mm / 1000.0;

        if (0 < pt_depth_m && pt_depth_m < DEPTH_MIN_DIST)
        {
            image.erase(iter);
            continue;
        }

        // std::find_if: http://c.biancheng.net/view/571.html
        int  feature_id = iter->first;
        auto it =
            find_if(feature.begin(), feature.end(),
                    [feature_id](const FeaturePerId &it) { return it.feature_id == feature_id; });

        if (it == feature.end())
        {
            if (has_semantic &&
                pt_depth_mm <
                    semantic_mask.at<unsigned short>((int)iter->second(4), (int)iter->second(3)) &&
                pt_depth_mm > 0)
            {
                image.erase(iter);
            }
            else
            {
                feature.emplace_back(FeaturePerId(feature_id, frame_count));
                feature.back().feature_per_frame.emplace_back(
                    FeaturePerFrame(iter->second, td, pt_depth_m));
            }
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.emplace_back(FeaturePerFrame(iter->second, td, pt_depth_m));
            last_track_num++;
            if (it->is_dynamic)
            {
                image.erase(iter);
            }
            else if (has_semantic &&
                     pt_depth_mm <
                         semantic_mask.at<unsigned short>(iter->second(4), iter->second(3)) &&
                     (pt_depth_mm != 0 ||
                      semantic_mask.at<unsigned short>(iter->second(4), iter->second(3)) == 0xffff))
            {
                it->is_dynamic = true;
                // it->feature_per_frame.emplace_back(FeaturePerFrame(iter->second, td, pt_depth_m));
                image.erase(iter);
            }
            // else
            // {
            //     it->feature_per_frame.emplace_back(FeaturePerFrame(iter->second, td, pt_depth_m));
            //     last_track_num++;
            // }
        }
    }
    has_semantic = false;
    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(!it.feature_per_frame.empty());
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ", j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l,
                                                                  int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int      idx_l = frame_count_l - it.start_frame;
            int      idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;

            corres.emplace_back(a, b);
        }
    }
    return corres;
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorrespondingWithDepth(int frame_count_l,
                                                                           int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int      idx_l = frame_count_l - it.start_frame;
            int      idx_r = frame_count_r - it.start_frame;

            double depth_a = it.feature_per_frame[idx_l].depth;
            if (depth_a < 0.1 || depth_a > 10)  // max and min measurement
                continue;
            double depth_b = it.feature_per_frame[idx_r].depth;
            if (depth_b < 0.1 || depth_b > 10)  // max and min measurement
                continue;
            a = it.feature_per_frame[idx_l].point;
            b = it.feature_per_frame[idx_r].point;
            a = a * depth_a;
            b = b * depth_b;

            corres.emplace_back(a, b);
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        if (it_per_id.is_dynamic)
        {
            continue;
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        // ROS_INFO("feature id %d , start_frame %d, depth %f ",
        // it_per_id->feature_id, it_per_id-> start_frame,
        // it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        if (it_per_id.is_dynamic)
        {
            continue;
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int      feature_index = -1;
    for (auto &it_per_id : feature)
    {
        if (it_per_id.is_dynamic)
        {
            continue;
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d _ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        // if (it_per_id.used_num < 4)
        //     continue;
        // if (!(it_per_id.used_num >depth_corner > 0)
        // continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int             svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d             t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d             R0 = Rs[imu_i] * _ric[0];
        P0.leftCols<3>()               = Eigen::Matrix3d::Identity();
        P0.rightCols<1>()              = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d             t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d             R1 = Rs[imu_j] * _ric[0];
            Eigen::Vector3d             t  = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d             R  = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>()      = R.transpose();
            P.rightCols<1>()     = -R.transpose() * t;
            Eigen::Vector3d f    = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V =
            Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        // it_per_id->estimated_depth = -b / A;
        // it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        // it_per_id->estimated_depth = INIT_DEPTH;
        it_per_id.estimate_flag = 2;
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
            it_per_id.estimate_flag   = 0;
        }
    }
}

void FeatureManager::triangulateWithDepth(Vector3d _Ps[], Vector3d _tic[], Matrix3d _ric[])
{
    for (auto &it_per_id : feature)
    {
        if (it_per_id.estimated_depth > 0)
            continue;
        if (it_per_id.is_dynamic)
        {
            continue;
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        int imu_i = it_per_id.start_frame;

        Eigen::Vector3d tr = _Ps[imu_i] + Rs[imu_i] * _tic[0];
        Eigen::Matrix3d Rr = Rs[imu_i] * _ric[0];

        vector<double> verified_depths;
        int            no_depth_num = 0;

        vector<double> rough_depths;
        /**
         * @brief
         * 对同一id特征点进行深度交叉验证，并将重投影误差小于阈值的点的，在初始帧上的深度做平均作为估计的深度
         */
        for (int k = 0; k < (int)it_per_id.feature_per_frame.size(); k++)
        {
            if (it_per_id.feature_per_frame[k].depth == 0)
            {
                no_depth_num++;
                continue;
            }
            Eigen::Vector3d t0 = _Ps[imu_i + k] + Rs[imu_i + k] * _tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i + k] * _ric[0];
            Eigen::Vector3d point0(it_per_id.feature_per_frame[k].point *
                                   it_per_id.feature_per_frame[k].depth);

            // transform to reference frame
            Eigen::Vector3d t2r = Rr.transpose() * (t0 - tr);
            Eigen::Matrix3d R2r = Rr.transpose() * R0;

            for (int j = 0; j < (int)it_per_id.feature_per_frame.size(); j++)
            {
                if (k == j)
                    continue;
                Eigen::Vector3d t1  = _Ps[imu_i + j] + Rs[imu_i + j] * _tic[0];
                Eigen::Matrix3d R1  = Rs[imu_i + j] * _ric[0];
                Eigen::Vector3d t20 = R0.transpose() * (t1 - t0);
                Eigen::Matrix3d R20 = R0.transpose() * R1;

                Eigen::Vector3d point1_projected = R20.transpose() * point0 - R20.transpose() * t20;
                Eigen::Vector2d point1_2d(it_per_id.feature_per_frame[j].point.x(),
                                          it_per_id.feature_per_frame[j].point.y());
                Eigen::Vector2d residual =
                    point1_2d - Vector2d(point1_projected.x() / point1_projected.z(),
                                         point1_projected.y() / point1_projected.z());
                if (residual.norm() < 10.0 / 460)
                {  // this can also be adjust to improve performance
                    Eigen::Vector3d point_r = R2r * point0 + t2r;

                    if (it_per_id.feature_per_frame[k].depth > DEPTH_MAX_DIST)
                    {
                        rough_depths.push_back(point_r.z());
                    }
                    else
                    {
                        verified_depths.push_back(point_r.z());
                    }
                }
            }
        }

        if (verified_depths.empty())
        {
            if (rough_depths.empty())
            {
                if (no_depth_num == it_per_id.feature_per_frame.size())
                {
                    int imu_j = imu_i - 1;

                    ROS_ASSERT(NUM_OF_CAM == 1);
                    Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
                    int             svd_idx = 0;

                    Eigen::Matrix<double, 3, 4> P0;
                    Eigen::Vector3d             t0 = _Ps[imu_i] + Rs[imu_i] * _tic[0];
                    Eigen::Matrix3d             R0 = Rs[imu_i] * ric[0];
                    P0.leftCols<3>()               = Eigen::Matrix3d::Identity();
                    P0.rightCols<1>()              = Eigen::Vector3d::Zero();

                    for (auto &it_per_frame : it_per_id.feature_per_frame)
                    {
                        imu_j++;

                        Eigen::Vector3d             t1 = _Ps[imu_j] + Rs[imu_j] * _tic[0];
                        Eigen::Matrix3d             R1 = Rs[imu_j] * ric[0];
                        Eigen::Vector3d             t  = R0.transpose() * (t1 - t0);
                        Eigen::Matrix3d             R  = R0.transpose() * R1;
                        Eigen::Matrix<double, 3, 4> P;
                        P.leftCols<3>()      = R.transpose();
                        P.rightCols<1>()     = -R.transpose() * t;
                        Eigen::Vector3d f    = it_per_frame.point.normalized();
                        svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
                        svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

                        if (imu_i == imu_j)
                            continue;
                    }
                    ROS_ASSERT(svd_idx == svd_A.rows());
                    Eigen::Vector4d svd_V =
                        Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV)
                            .matrixV()
                            .rightCols<1>();
                    double svd_method = svd_V[2] / svd_V[3];

                    if (svd_method < DEPTH_MIN_DIST)
                    {
                        it_per_id.estimated_depth = DEPTH_MAX_DIST;
                        it_per_id.estimate_flag   = 2;
                    }
                    else
                    {
                        it_per_id.estimated_depth = svd_method;
                        it_per_id.estimate_flag   = 2;
                    }
                }
                else
                {
                    continue;
                }
            }
            else
            {
                double depth_sum =
                    std::accumulate(std::begin(rough_depths), std::end(rough_depths), 0.0);
                double depth_ave          = depth_sum / rough_depths.size();
                it_per_id.estimated_depth = depth_ave;
                it_per_id.estimate_flag   = 0;
            }
        }
        else
        {
            double depth_sum =
                std::accumulate(std::begin(verified_depths), std::end(verified_depths), 0.0);
            double depth_ave          = depth_sum / verified_depths.size();
            it_per_id.estimated_depth = depth_ave;
            it_per_id.estimate_flag   = 1;
        }

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
            it_per_id.estimate_flag   = 0;
        }
    }
}

bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P,
                                    vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w
    R_initial = R.inverse();       // R_c_w  from world to cam tzhang
    P_initial = -(R_initial * P);  // P_c_w

    // printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);  //旋转矩阵到旋转向量 tzhang
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool    pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    // pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 /
    // focalLength, 0.99, inliers);

    if (!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r);  //旋转向量到旋转矩阵 tzhang
    // cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();  // R_w_c
    P = R * (-T_pnp);

    return true;
}

void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[],
                                        Matrix3d ric[])
{
    if (frameCnt >
        0)  //对第一帧图像不做处理；因为此时路标点还未三角化，需要利用第一帧双目图像，进行路标点三角化
            // tzhang
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        for (auto &it_per_id : feature)  //遍历每个路标点 tzhang
        {
            if (it_per_id.estimated_depth >
                0)  //该路标点完成了初始化，使用初始化完成的路标点，获取3D-2D点对
            // tzhang
            {
                int index = frameCnt - it_per_id.start_frame;
                if ((int)it_per_id.feature_per_frame.size() >= index + 1)  // tzhang
                    // 该路标点从start_frame图像帧到frameCnt对应的图像帧都能被观测到
                {
                    //路标点在IMU坐标系坐标，第一次看到该路标点的图像帧时刻
                    // tzhang
                    Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point *
                                                  it_per_id.estimated_depth) +
                                        tic[0];
                    // 路标点在世界坐标系下的坐标
                    Vector3d ptsInWorld =
                        Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame];

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(),
                                        ptsInWorld.z());  //世界坐标系下三维坐标 tzhang
                    cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(),
                                        it_per_id.feature_per_frame[index].point.y());
                    pts3D.push_back(point3d);
                    pts2D.push_back(point2d);
                }
            }
        }
        Eigen::Matrix3d RCam;
        Eigen::Vector3d PCam;
        // trans to w_T_cam
        // 以上一帧图像与世界坐标系之间的位姿变换作为后续PnP求解的初值 tzhang
        RCam = Rs[frameCnt - 1] * ric[0];  // R_w_c
        PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];

        if (solvePoseByPnP(RCam, PCam, pts2D, pts3D))
        {
            // trans to w_T_imu
            Rs[frameCnt] = RCam * ric[0].transpose();  // R_w_i = R_w_c*R_c_i
            Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;
        }
    }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        int index = it->feature_id;
        itSet     = outlierIndex.find(index);
        if (itSet != outlierIndex.end())
        {
            feature.erase(it);
            // printf("remove outlier %d \n", index);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P,
                                          Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i   = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j   = new_R.transpose() * (w_pts_i - new_P);
                double          dep_j   = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i =
        it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j =
        it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double   ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    // int r_i = frame_count - 2;
    // int r_j = frame_count - 1;
    // p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] *
    // ric[camera_id_i] * p_i;
    p_i_comp     = p_i;
    double dep_i = p_i(2);
    double u_i   = p_i(0) / dep_i;
    double v_i   = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp   = p_i_comp(0) / dep_i_comp;
    double v_i_comp   = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
