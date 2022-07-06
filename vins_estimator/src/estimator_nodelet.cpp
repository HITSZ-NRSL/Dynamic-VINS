#include <condition_variable>
#include <cv_bridge/cv_bridge.h>
#include <map>
#include <mutex>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <ros/ros.h>
#include <set>
#include <string>
#include <thread>

#include "estimator/estimator.h"
#include "feature_tracker/feature_tracker.h"
#include "ros/console_backend.h"
#include "sensor_msgs/image_encodings.h"
#include "utility/parameters.h"
#include "utility/tic_toc.h"
#include "utility/visualization.h"

#include <yolo_ros/DetectionMessages.h>

#include <nodelet/nodelet.h>  // 基类Nodelet所在的头文件
#include <pluginlib/class_list_macros.h>

namespace estimator_nodelet_ns
{
class EstimatorNodelet : public nodelet::Nodelet  //任何nodelet plugin都要继承Nodelet类。
{
public:
    EstimatorNodelet() = default;

private:
    void onInit() override
    {
        ros::NodeHandle &pn = getPrivateNodeHandle();
        ros::NodeHandle &nh = getMTNodeHandle();

        ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

        readParameters(pn);

        estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
        ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
        ROS_WARN("waiting for image, semantic and imu...");

        registerPub(nh);

        sub_image = nh.subscribe(IMAGE_TOPIC, 100, &EstimatorNodelet::image_callback, this);
        sub_depth = nh.subscribe(DEPTH_TOPIC, 100, &EstimatorNodelet::depth_callback, this);

        sub_semantic =
            nh.subscribe("/untracked_info", 100, &EstimatorNodelet::semantic_callback, this);
        if (USE_IMU)
            sub_imu = nh.subscribe(IMU_TOPIC, 100, &EstimatorNodelet::imu_callback, this,
                                   ros::TransportHints().tcpNoDelay());
        // topic from pose_graph, notify if there's relocalization
        sub_relo_points = nh.subscribe("/pose_graph/match_points", 10,
                                       &EstimatorNodelet::relocalization_callback, this);

        dura = std::chrono::milliseconds(2);

        trackThread   = std::thread(&EstimatorNodelet::process_tracker, this);
        processThread = std::thread(&EstimatorNodelet::process, this);
    }

    Estimator estimator;

    // thread relevance
    std::thread               trackThread, processThread;
    std::chrono::milliseconds dura;
    std::condition_variable   con_tracker;
    std::condition_variable   con_estimator;
    std::mutex                m_feature;
    std::mutex                m_backend;
    std::mutex                m_buf;
    std::mutex                m_vis;

    // ROS and data buf relevance
    ros::Subscriber                   sub_semantic, sub_imu, sub_relo_points, sub_image, sub_depth;
    queue<sensor_msgs::ImageConstPtr> img_buf;
    queue<sensor_msgs::ImageConstPtr> depth_buf;
    queue<pair<pair<std_msgs::Header, sensor_msgs::ImageConstPtr>,
               map<int, Eigen::Matrix<double, 7, 1>>>>
                                               feature_buf;
    queue<yolo_ros::DetectionMessagesConstPtr> semantic_buf;
    queue<sensor_msgs::PointCloudConstPtr>     relo_buf;
    queue<pair<std_msgs::Header, cv::Mat>>     vis_img_buf;

    bool init_feature = false;
    bool init_pub     = false;

    // frequency control relevance
    bool   first_image_flag = true;
    double first_image_time = 0;
    double last_image_time  = 0;
    int    pub_count        = 1;
    int    input_count      = 0;

    double last_imu_t = 0;

    void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
    {
        /**
         * @brief nullptr may cause crash
         * typename boost::detail::sp_member_access<T>::type = const
         * sensor_msgs::Imu_<std::allocator<void> >*]: Assertion `px != 0' failed.
         * https://github.com/mavlink/mavros/issues/432
         * https://github.com/mavlink/mavros/pull/434
         */
        if (imu_msg)
        {
            if (imu_msg->header.stamp.toSec() <= last_imu_t)
            {
                ROS_WARN("imu message in disorder! %f", imu_msg->header.stamp.toSec());
                return;
            }

            last_imu_t = imu_msg->header.stamp.toSec();
            Vector3d acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y,
                         imu_msg->linear_acceleration.z);
            Vector3d gyr(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y,
                         imu_msg->angular_velocity.z);
            estimator.inputIMU(last_imu_t, acc, gyr);
        }
    }

    void image_callback(const sensor_msgs::ImageConstPtr &color_msg)
    {
        m_buf.lock();
        img_buf.emplace(color_msg);
        m_buf.unlock();
        con_tracker.notify_one();
    }

    void depth_callback(const sensor_msgs::ImageConstPtr &depth_msg)
    {
        m_buf.lock();
        depth_buf.emplace(depth_msg);
        m_buf.unlock();
        con_tracker.notify_one();
    }

    void semantic_callback(const yolo_ros::DetectionMessagesConstPtr &semantic_msg)
    {
        m_buf.lock();
        semantic_buf.push(semantic_msg);
        m_buf.unlock();
    }

    void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
    {
        m_buf.lock();
        relo_buf.push(points_msg);
        m_buf.unlock();
    }

    void visualizeFeatureFilter(const map<int, Eigen::Matrix<double, 7, 1>> &features,
                                const yolo_ros::DetectionMessagesConstPtr   &semantic_msg)
    {
        cv::Mat vis_img;
        m_vis.lock();
        while (!vis_img_buf.empty())
        {
            if (vis_img_buf.front().first.stamp.toSec() == semantic_msg->header.stamp.toSec())
            {
                vis_img = vis_img_buf.front().second;
                vis_img_buf.pop();
                break;
            }
            else if (vis_img_buf.front().first.stamp.toSec() < semantic_msg->header.stamp.toSec())
            {
                vis_img_buf.pop();
            }
            else
            {
                m_vis.unlock();
                return;
            }
        }
        m_vis.unlock();

        // Show image with tracked points in rviz (by topic pub_match)
        for (auto &feature : features)
        {
            cv::circle(vis_img, cv::Point(feature.second[3], feature.second[4]), 5,
                       cv::Scalar(0, 255, 255), 2);
        }
        for (auto &object : semantic_msg->data)
        {
            if (std::find(DYNAMIC_LABEL.begin(), DYNAMIC_LABEL.end(), object.label) !=
                DYNAMIC_LABEL.end())
            {
                cv::rectangle(vis_img, cv::Point(object.x1, object.y1),
                              cv::Point(object.x2, object.y2), cv::Scalar(0, 255, 0), 2);
            }
        }
        pubTrackImg(vis_img);

        // cv::imshow("grids_detector_img",
        //            estimator.featureTracker.grids_detector_img);
        // cv::moveWindow("grids_detector_img", 0, 0);
        // cv::waitKey(1);

        // cv::imshow("feature_img", vis_img);
        // cv::moveWindow("feature_img", 0, 0);
        // cv::waitKey(1);

        cv::Mat show_semantic_mask = estimator.f_manager.semantic_mask.clone();
        for (int i = 0; i < COL - 1; ++i)
        {
            for (int k = 0; k < ROW - 1; ++k)
            {
                /* if (show_semantic_mask.at<unsigned short>(k, i) > 0)
                  show_semantic_mask.at<unsigned short>(k, i) = 0xffff; */
                if (estimator.f_manager.semantic_mask.at<unsigned short>(k, i) >
                        estimator.f_manager.depth_img.at<unsigned short>(k, i) &&
                    estimator.f_manager.depth_img.at<unsigned short>(k, i) > 0)
                {
                    show_semantic_mask.at<unsigned short>(k, i) = 0xffff;
                }
            }
        }
        pubSemanticMask(show_semantic_mask);

        // cv::imshow("semantic_mask_h", show_semantic_mask);
        // cv::moveWindow("semantic_mask_h", COL, 0);
        // cv::waitKey(1);
        // cv::imshow("semantic_mask_v", show_semantic_mask);
        // cv::moveWindow("semantic_mask_v", 0, ROW);
        // cv::waitKey(1);
    }

    bool semanticAvailable(double t)
    {
        if (!semantic_buf.empty() && t <= semantic_buf.back()->header.stamp.toSec())
            return true;
        else
            return false;
    }

    // thread: feature tracker
    [[noreturn]] void process_tracker()
    {
        while (1)
        {
            {
                sensor_msgs::ImageConstPtr color_msg = nullptr;
                sensor_msgs::ImageConstPtr depth_msg = nullptr;

                std::unique_lock<std::mutex> locker(m_buf);
                while (img_buf.empty() || depth_buf.empty())
                {
                    con_tracker.wait(locker);
                }

                double time_color = img_buf.front()->header.stamp.toSec();
                double time_depth = depth_buf.front()->header.stamp.toSec();

                if (time_color < time_depth - 0.003)
                {
                    img_buf.pop();
                    ROS_DEBUG("throw color\n");
                }
                else if (time_color > time_depth + 0.003)
                {
                    depth_buf.pop();
                    ROS_DEBUG("throw depth\n");
                }
                else
                {
                    color_msg = img_buf.front();
                    img_buf.pop();
                    depth_msg = depth_buf.front();
                    depth_buf.pop();
                }
                locker.unlock();

                if (color_msg == nullptr || depth_msg == nullptr)
                {
                    ROS_DEBUG("time_color = %f, time_depth = %f\n", time_color, time_depth);
                    continue;
                }

                if (first_image_flag)
                {
                    first_image_flag = false;
                    first_image_time = time_color;
                    last_image_time  = time_color;
                    continue;
                }

                // detect unstable camera stream
                if (time_color - last_image_time > 1.0 || time_color < last_image_time)
                {
                    ROS_WARN("image discontinue! reset the feature tracker!");
                    first_image_flag = true;
                    last_image_time  = 0;
                    pub_count        = 1;

                    ROS_WARN("restart the estimator!");
                    m_feature.lock();
                    while (!feature_buf.empty())
                        feature_buf.pop();
                    m_feature.unlock();
                    m_backend.lock();
                    estimator.clearState();
                    estimator.setParameter();
                    m_backend.unlock();
                    last_imu_t = 0;

                    continue;
                }

                // frequency control
                if (round(1.0 * input_count / (time_color - first_image_time)) > FRONTEND_FREQ)
                {
                    ROS_DEBUG("Skip this frame.%f",
                              1.0 * input_count / (time_color - first_image_time));
                    continue;
                }
                ++input_count;

                // frequency control
                if (round(1.0 * pub_count / (time_color - first_image_time)) <= FREQ)
                {
                    PUB_THIS_FRAME = true;
                    // reset the frequency control
                    if (abs(1.0 * pub_count / (time_color - first_image_time) - FREQ) < 0.01 * FREQ)
                    {
                        first_image_time = time_color;
                        pub_count        = 0;
                        input_count      = 0;
                    }
                }
                else
                    PUB_THIS_FRAME = false;

                TicToc t_r;
                // encodings in ros:
                // http://docs.ros.org/diamondback/api/sensor_msgs/html/image__encodings_8cpp_source.html
                // color has encoding RGB8
                cv_bridge::CvImageConstPtr ptr;
                if (color_msg->encoding ==
                    "8UC1")  // shan:why 8UC1 need this operation? Find
                             // answer:https://github.com/ros-perception/vision_opencv/issues/175
                {
                    sensor_msgs::Image img;
                    img.header       = color_msg->header;
                    img.height       = color_msg->height;
                    img.width        = color_msg->width;
                    img.is_bigendian = color_msg->is_bigendian;
                    img.step         = color_msg->step;
                    img.data         = color_msg->data;
                    img.encoding     = "mono8";
                    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
                }
                else
                    ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::MONO8);

                if (USE_IMU)
                {
                    Matrix3d &&relative_R =
                        estimator.predictMotion(last_image_time, time_color + estimator.td);
                    estimator.featureTracker.readImage(ptr->image, time_color, relative_R);
                }
                else
                    estimator.featureTracker.readImage(ptr->image, time_color);

                last_image_time = time_color;
                // always 0

                // update all id in ids[]
                // If has ids[i] == -1 (newly added pts by cv::goodFeaturesToTrack),
                // substitute by gloabl id counter (n_id)
                for (unsigned int i = 0;; i++)
                {
                    bool completed = false;
                    completed |= estimator.featureTracker.updateID(i);
                    if (!completed)
                        break;
                }
                if (PUB_THIS_FRAME)
                {
                    pub_count++;

                    std_msgs::Header                      feature_header = color_msg->header;
                    map<int, Eigen::Matrix<double, 7, 1>> image;
                    auto &un_pts       = estimator.featureTracker.cur_un_pts;
                    auto &cur_pts      = estimator.featureTracker.cur_pts;
                    auto &ids          = estimator.featureTracker.ids;
                    auto &pts_velocity = estimator.featureTracker.pts_velocity;
                    for (unsigned int j = 0; j < ids.size(); j++)
                    {
                        if (estimator.featureTracker.track_cnt[j] > 1)
                        {
                            int                    p_id = ids[j];
                            geometry_msgs::Point32 p;
                            double                 x = un_pts[j].x;
                            double                 y = un_pts[j].y;
                            double                 z = 1;

                            int    v          = p_id * NUM_OF_CAM + 0.5;
                            int    feature_id = v / NUM_OF_CAM;
                            double p_u        = cur_pts[j].x;
                            double p_v        = cur_pts[j].y;
                            double velocity_x = pts_velocity[j].x;
                            double velocity_y = pts_velocity[j].y;

                            ROS_ASSERT(z == 1);
                            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                            image[feature_id] = xyz_uv_velocity;
                        }
                    }

                    if (!init_pub)
                    {
                        init_pub = true;
                    }
                    else
                    {
                        if (!init_feature)
                        {
                            // skip the first detected feature, which doesn't contain optical
                            // flow speed
                            init_feature = true;
                            continue;
                        }
                        if (!image.empty())
                        {
                            m_feature.lock();
                            feature_buf.push(
                                make_pair(make_pair(feature_header, depth_msg), std::move(image)));
                            m_feature.unlock();
                            con_estimator.notify_one();
                        }
                        else
                        {
                            first_image_time = time_color;
                            pub_count        = 0;
                            input_count      = 0;
                            continue;
                        }
                    }

                    // Show image with tracked points in rviz (by topic pub_match)
                    if (SHOW_TRACK)
                    {
                        cv::Mat show_img = ptr->image;
                        ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
                        cv::Mat stereo_img = ptr->image;
                        cv::Mat tmp_img    = stereo_img.rowRange(0, ROW);
                        cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                        for (unsigned int j = 0; j < estimator.featureTracker.cur_pts.size(); j++)
                        {
                            if (estimator.featureTracker.track_cnt[j] > 1)
                            {
                                double len = std::min(
                                    1.0, 1.0 * estimator.featureTracker.track_cnt[j] / WINDOW_SIZE);
                                cv::circle(tmp_img, estimator.featureTracker.cur_pts[j], 5,
                                           cv::Scalar(255 * (1 - len), 0, 255 * len), -1);
                                // draw speed line
                                //                     Vector2d tmp_cur_un_pts
                                //                     (trackerData[i].cur_un_pts[j].x,
                                //                     trackerData[i].cur_un_pts[j].y); Vector2d
                                //                     tmp_pts_velocity
                                //                     (trackerData[i].pts_velocity[j].x,
                                //                     trackerData[i].pts_velocity[j].y); Vector3d
                                //                     tmp_prev_un_pts; tmp_prev_un_pts.head(2) =
                                //                     tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                                //                     tmp_prev_un_pts.z() = 1;
                                //                     Vector2d tmp_prev_uv;
                                //                     trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts,
                                //                     tmp_prev_uv); cv::line(tmp_img,
                                //                     trackerData[i].cur_pts[j],
                                //                     cv::Point2f(tmp_prev_uv.x(),
                                //                     tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1
                                //                     , 8, 0);

                                // char name[10];
                                // sprintf(name, "%d", trackerData[i].ids[j]);
                                // cv::putText(tmp_img, name, trackerData[i].cur_pts[j],
                                // cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                            }
                        }
                        if (USE_IMU)
                        {
                            for (auto &predict_pt : estimator.featureTracker.predict_pts)
                            {
                                cv::circle(tmp_img, predict_pt, 2, cv::Scalar(0, 255, 0), -1);
                            }
                        }

                        m_vis.lock();
                        vis_img_buf.push(make_pair(feature_header, tmp_img));
                        m_vis.unlock();
                    }
                }
                static double whole_process_time = 0;
                static size_t cnt_frame          = 0;
                ++cnt_frame;
                double per_process_time = t_r.toc();
                whole_process_time += per_process_time;
                ROS_DEBUG("average feature tracking costs: %f", whole_process_time / cnt_frame);
                // ROS_DEBUG("feature tracking costs: %f", per_process_time);
            }
            std::this_thread::sleep_for(dura);
        }
    }

    // thread: visual-inertial odometry
    [[noreturn]] void process()
    {
        while (true)
        {
            std::unique_lock<std::mutex> locker(m_feature);
            while (feature_buf.empty())
            {
                con_estimator.wait(locker);
            }

            pair<pair<std_msgs::Header, sensor_msgs::ImageConstPtr>,
                 map<int, Eigen::Matrix<double, 7, 1>>>
                feature_msg(std::move(feature_buf.front()));
            feature_buf.pop();
            locker.unlock();

            TicToc t_backend;
            m_backend.lock();
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = nullptr;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }

            if (relo_msg != nullptr)
            {
                vector<Vector3d> match_points;
                double           frame_stamp = relo_msg->header.stamp.toSec();
                for (auto point : relo_msg->points)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = point.x;
                    u_v_id.y() = point.y;
                    u_v_id.z() = point.z;
                    match_points.push_back(u_v_id);
                }
                Vector3d    relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1],
                                   relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4],
                                   relo_msg->channels[0].values[5],
                                   relo_msg->channels[0].values[6]);
                Matrix3d    relo_r = relo_q.toRotationMatrix();
                int         frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            // depth has encoding TYPE_16UC1
            cv::Mat depth_img;
            if (feature_msg.first.second->encoding == "mono16" ||
                feature_msg.first.second->encoding == "16UC1")
            {
                depth_img = cv_bridge::toCvShare(feature_msg.first.second)->image;
            }
            else if (feature_msg.first.second->encoding == "32FC1")
            {
                cv::Mat depth_32fc1 = cv_bridge::toCvShare(feature_msg.first.second)->image;
                depth_32fc1.convertTo(depth_img, CV_16UC1, 1000);
            }
            else
            {
                ROS_ASSERT_MSG(1, "Unknown depth encoding!");
            }
            estimator.f_manager.inputDepth(depth_img);

            double feature_time = feature_msg.first.first.stamp.toSec();
            while (!semanticAvailable(feature_time))
            {
                // printf("waiting for semantic info ... \r");
                std::this_thread::sleep_for(dura);
            }

            yolo_ros::DetectionMessagesConstPtr semantic_msg = nullptr;
            m_buf.lock();
            while (!semantic_buf.empty())
            {
                if (semantic_buf.front()->header.stamp.toSec() < feature_time)
                {
                    semantic_buf.pop();
                }
                else if (semantic_buf.front()->header.stamp.toSec() == feature_time)
                {
                    semantic_msg = semantic_buf.front();
                    semantic_buf.pop();
                    break;
                }
                else
                {
                    break;
                }
            }
            m_buf.unlock();

            TicToc semantic_time;
            if (semantic_msg != nullptr)
            {
                for (auto &object : semantic_msg->data)
                {
                    if (std::find(DYNAMIC_LABEL.begin(), DYNAMIC_LABEL.end(), object.label) !=
                        DYNAMIC_LABEL.end())
                    {
                        // type: uint32
                        int x1 = ((int)object.x1 - 10 > 0) ? ((int)object.x1 - 10) : 0;
                        int y1 = ((int)object.y1 - 10 > 0) ? ((int)object.y1 - 10) : 0;
                        int x2 = ((int)object.x2 + 10 < COL) ? ((int)object.x2 + 10) : COL - 1;
                        int y2 = ((int)object.y2 + 10 < ROW) ? ((int)object.y2 + 10) : ROW - 1;
                        if (x2 >= 0 && x1 < COL && y2 >= 0 && y1 < ROW)
                            estimator.f_manager.setSemanticMask(x1, y1, x2, y2);
                    }
                }
            }

            static double whole_semantic_time = 0;
            whole_semantic_time += semantic_time.toc();
            TicToc cmd_time;
            estimator.f_manager.compensateMissedDetection();

            static double whole_cmd_time = 0;
            static size_t cnt_frame      = 0;
            ++cnt_frame;
            whole_cmd_time += cmd_time.toc();
            ROS_DEBUG("average cmd costs: %f", whole_cmd_time / cnt_frame);
            ROS_DEBUG("average semantic_time costs: %f", whole_semantic_time / cnt_frame);

            TicToc t_processImage;
            estimator.processImage(feature_msg.second, feature_msg.first.first);

            std_msgs::Header header = feature_msg.first.first;
            header.frame_id         = "map";
            pubOdometry(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != nullptr)
                pubRelocalization(estimator);

            m_backend.unlock();
            if (SHOW_TRACK && semantic_msg != nullptr)
            {
                pubKeyPoses(estimator, header);
                pubCameraPose(estimator, header);
                pubPointCloud(estimator, header);
                visualizeFeatureFilter(feature_msg.second, semantic_msg);
            }

            static double whole_process_time = 0;
            double        per_process_time   = t_backend.toc();
            whole_process_time += per_process_time;
            printStatistics(estimator, per_process_time);
            ROS_DEBUG("average backend costs: %f", whole_process_time / cnt_frame);
            // ROS_DEBUG("backend costs: %f", per_process_time);
            std::this_thread::sleep_for(dura);
        }
    }
};

PLUGINLIB_EXPORT_CLASS(estimator_nodelet_ns::EstimatorNodelet, nodelet::Nodelet)
}  // namespace estimator_nodelet_ns