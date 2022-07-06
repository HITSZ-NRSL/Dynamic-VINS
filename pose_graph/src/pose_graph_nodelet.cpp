#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <vector>

#include <message_filters/synchronizer.h>

#include "keyframe/keyframe.h"
#include "pose_graph/pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <queue>
#include <ros/package.h>
#include <thread>
#include <visualization_msgs/Marker.h>

#include <nodelet/nodelet.h> // 基类Nodelet所在的头文件
#include <pluginlib/class_list_macros.h>

#define SKIP_FIRST_CNT 10

camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
Eigen::Matrix<double, 3, 1> ti_d;
Eigen::Matrix<double, 3, 3> qi_d;
ros::Publisher pub_match_img;
ros::Publisher pub_match_points;
int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
int ROW;
int COL;
std::string VINS_RESULT_PATH;
int DEBUG_IMAGE;
int FAST_RELOCALIZATION;
namespace pose_graph_nodelet_ns {
class PoseGraphNodelet
    : public nodelet::Nodelet //任何nodelet plugin都要继承Nodelet类。
{
public:
  PoseGraphNodelet() {
    frame_index = 0;
    sequence = 1;
    skip_first_cnt = 0;
    skip_cnt = 0;
    load_flag = 0;
    start_flag = 0;
    SKIP_DIS = 0;

    last_image_time = -1;

    last_t = Eigen::Vector3d(-100, -100, -100);

    cameraposevisual = new CameraPoseVisualization(1, 0, 0, 1);
  }

private:
  void onInit() override {

    ros::NodeHandle &pn = getPrivateNodeHandle();
    ros::NodeHandle &nh = getMTNodeHandle();
    posegraph.registerPub(nh);

    // read param
    pn.getParam("visualization_shift_x", VISUALIZATION_SHIFT_X);
    pn.getParam("visualization_shift_y", VISUALIZATION_SHIFT_Y);
    pn.getParam("skip_cnt", SKIP_CNT);
    pn.getParam("skip_dis", SKIP_DIS);
    std::string config_file;
    pn.getParam("config_file", config_file);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
      std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    double camera_visual_size = fsSettings["visualize_camera_size"];
    cameraposevisual->setScale(camera_visual_size);
    cameraposevisual->setLineWidth(camera_visual_size / 10.0);

    LOOP_CLOSURE = fsSettings["loop_closure"];
    std::string IMAGE_TOPIC;
    int LOAD_PREVIOUS_POSE_GRAPH;
    // prepare for loop closure (load vocabulary, set topic, etc)
    if (LOOP_CLOSURE) {
      ROW = fsSettings["image_height"];
      COL = fsSettings["image_width"];

      std::string pkg_path = ros::package::getPath("pose_graph");
      string vocabulary_file = pkg_path + "/../support_files/brief_k10L6.bin";
      cout << "vocabulary_file" << vocabulary_file << endl;
      posegraph.loadVocabulary(vocabulary_file);

      BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";
      cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;
      m_camera =
          camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
              config_file.c_str());

      fsSettings["image_topic"] >> IMAGE_TOPIC;
      fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
      fsSettings["output_path"] >> VINS_RESULT_PATH;
      fsSettings["save_image"] >> DEBUG_IMAGE;

      cv::Mat cv_qid, cv_tid;
      fsSettings["extrinsicRotation"] >> cv_qid;
      fsSettings["extrinsicTranslation"] >> cv_tid;
      cv::cv2eigen(cv_qid, qi_d);
      cv::cv2eigen(cv_tid, ti_d);

      int USE_IMU = fsSettings["imu"];
      posegraph.setIMUFlag(USE_IMU);

      VISUALIZE_IMU_FORWARD = fsSettings["visualize_imu_forward"];
      LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];
      FAST_RELOCALIZATION = fsSettings["fast_relocalization"];
      VINS_RESULT_PATH = VINS_RESULT_PATH + "/vins_result_loop.csv";
      std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
      fout.close();
      fsSettings.release();
      // not used
      if (LOAD_PREVIOUS_POSE_GRAPH) {
        printf("load pose graph\n");
        m_process.lock();
        posegraph.loadPoseGraph();
        m_process.unlock();
        printf("load pose graph finish\n");
        load_flag = true;
      } else {
        printf("no previous pose graph\n");
        load_flag = true;
      }
    }

    fsSettings.release();
    // publish camera pose by imu propagate and odometry (Ps and Rs of curr
    // frame) not important

    sub_imu_forward =
        nh.subscribe("/vins_estimator/imu_propagate", 100,
                     &PoseGraphNodelet::imu_forward_callback, this);
    // odometry_buf
    sub_vio = nh.subscribe("/vins_estimator/odometry", 100,
                           &PoseGraphNodelet::vio_callback, this);

    sub_image =
        nh.subscribe(IMAGE_TOPIC, 100, &PoseGraphNodelet::image_callback, this);

    // get keyframe_pose(Ps and Rs), store in pose_buf (marginalization_flag ==
    // 0)
    sub_pose = nh.subscribe("/vins_estimator/keyframe_pose", 100,
                            &PoseGraphNodelet::pose_callback, this);
    // get extrinsic (ric qic  odometry.pose.pose.position and
    // odometry.pose.pose.orientation) update tic and qic real-time
    sub_extrinsic = nh.subscribe("/vins_estimator/extrinsic", 100,
                                 &PoseGraphNodelet::extrinsic_callback, this);
    // get keyframe_point(pointclude), store in point_buf (marginalization_flag
    // == 0)
    sub_point = nh.subscribe("/vins_estimator/keyframe_point", 100,
                             &PoseGraphNodelet::point_callback, this);

    // do relocalization here.
    // pose_graph publish match_points to vins_estimator, estimator then publish
    // relo_relative_pose
    sub_relo_relative_pose =
        nh.subscribe("/vins_estimator/relo_relative_pose", 100,
                     &PoseGraphNodelet::relo_relative_pose_callback, this);

    pub_match_img = nh.advertise<sensor_msgs::Image>("match_image", 100);
    pub_camera_pose_visual = nh.advertise<visualization_msgs::MarkerArray>(
        "/pose_graph/camera_pose_visual", 100);
    pub_key_odometrys = nh.advertise<visualization_msgs::Marker>(
        "/pose_graph/key_odometrys", 100);
    // not used
    pub_vio_path =
        nh.advertise<nav_msgs::Path>("/pose_graph/no_loop_path", 100);
    pub_match_points =
        nh.advertise<sensor_msgs::PointCloud>("/pose_graph/match_points", 100);

    // main thread
    measurement_process = std::thread(&PoseGraphNodelet::process, this);
    // not used
    keyboard_command_process = std::thread(&PoseGraphNodelet::command, this);
  }

  ros::Subscriber sub_image, sub_imu_forward, sub_vio;
  ros::Subscriber sub_pose, sub_extrinsic, sub_point, sub_relo_relative_pose;

  std::thread measurement_process, keyboard_command_process;

  queue<sensor_msgs::ImageConstPtr> image_buf;
  queue<sensor_msgs::PointCloudConstPtr> point_buf;
  queue<nav_msgs::Odometry::ConstPtr> pose_buf;
  queue<Eigen::Vector3d> odometry_buf;
  std::mutex m_buf;
  std::mutex m_process;
  int frame_index;
  int sequence;
  PoseGraph posegraph;
  int skip_first_cnt;
  int SKIP_CNT{};
  int skip_cnt;
  bool load_flag;
  bool start_flag;
  double SKIP_DIS;

  int VISUALIZE_IMU_FORWARD{};
  int LOOP_CLOSURE{};

  ros::Publisher pub_camera_pose_visual;
  ros::Publisher pub_key_odometrys;
  ros::Publisher pub_vio_path;
  nav_msgs::Path no_loop_path;

  CameraPoseVisualization *cameraposevisual;

  Eigen::Vector3d last_t; //(-100, -100, -100);
  double last_image_time;

  // not used in my case, just ignore sequence 1-5
  void new_sequence() {
    printf("new sequence\n");
    sequence++;
    printf("sequence cnt %d \n", sequence);
    if (sequence > 5) {
      ROS_WARN("only support 5 sequences since it's boring to copy code for "
               "more sequences.");
      ROS_BREAK();
    }
    posegraph.posegraph_visualization->reset();
    posegraph.publish();
    m_buf.lock();
    while (!image_buf.empty())
      image_buf.pop();
    while (!point_buf.empty())
      point_buf.pop();
    while (!pose_buf.empty())
      pose_buf.pop();
    while (!odometry_buf.empty())
      odometry_buf.pop();
    m_buf.unlock();
  }

  void image_callback(const sensor_msgs::ImageConstPtr &image_msg) {
    // ROS_WARN("image_callback!");
    if (!LOOP_CLOSURE)
      return;
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
    // printf(" image time %f \n", image_msg->header.stamp.toSec());

    // detect unstable camera stream
    if (last_image_time == -1)
      last_image_time = image_msg->header.stamp.toSec();
    else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 ||
             image_msg->header.stamp.toSec() < last_image_time) {
      ROS_WARN("image discontinue! detect a new sequence!");
      new_sequence();
    }
    last_image_time = image_msg->header.stamp.toSec();
  }

  void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg) {
    // ROS_INFO("point_callback!");
    if (!LOOP_CLOSURE)
      return;
    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
    /*
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        printf("%d, 3D point: %f, %f, %f 2D point %f, %f \n",i ,
    point_msg->points[i].x, point_msg->points[i].y, point_msg->points[i].z,
                                                     point_msg->channels[i].values[0],
                                                     point_msg->channels[i].values[1]);
    }
    */
  }

  void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
    // ROS_INFO("pose_callback!");
    if (!LOOP_CLOSURE)
      return;
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
    /*
    printf("pose t: %f, %f, %f   q: %f, %f, %f %f \n",
    pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                                                       pose_msg->pose.pose.position.z,
                                                       pose_msg->pose.pose.orientation.w,
                                                       pose_msg->pose.pose.orientation.x,
                                                       pose_msg->pose.pose.orientation.y,
                                                       pose_msg->pose.pose.orientation.z);
    */
  }

  // not used
  void imu_forward_callback(const nav_msgs::Odometry::ConstPtr &forward_msg) {
    if (VISUALIZE_IMU_FORWARD) {
      Vector3d vio_t(forward_msg->pose.pose.position.x,
                     forward_msg->pose.pose.position.y,
                     forward_msg->pose.pose.position.z);
      Quaterniond vio_q;
      vio_q.w() = forward_msg->pose.pose.orientation.w;
      vio_q.x() = forward_msg->pose.pose.orientation.x;
      vio_q.y() = forward_msg->pose.pose.orientation.y;
      vio_q.z() = forward_msg->pose.pose.orientation.z;

      vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
      vio_q = posegraph.w_r_vio * vio_q;

      vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
      vio_q = posegraph.r_drift * vio_q;

      Vector3d vio_t_cam;
      Quaterniond vio_q_cam;
      vio_t_cam = vio_t + vio_q * tic;
      vio_q_cam = vio_q * qic;

      cameraposevisual->reset();
      cameraposevisual->add_pose(vio_t_cam, vio_q_cam);
      cameraposevisual->publish_by(pub_camera_pose_visual, forward_msg->header);
    }
  }

  void
  relo_relative_pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
    Vector3d relative_t =
        Vector3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                 pose_msg->pose.pose.position.z);
    Quaterniond relative_q;
    relative_q.w() = pose_msg->pose.pose.orientation.w;
    relative_q.x() = pose_msg->pose.pose.orientation.x;
    relative_q.y() = pose_msg->pose.pose.orientation.y;
    relative_q.z() = pose_msg->pose.pose.orientation.z;
    double relative_yaw = pose_msg->twist.twist.linear.x;
    int index = pose_msg->twist.twist.linear.y;
    // printf("receive index %d \n", index );
    Eigen::Matrix<double, 8, 1> loop_info;
    loop_info << relative_t.x(), relative_t.y(), relative_t.z(), relative_q.w(),
        relative_q.x(), relative_q.y(), relative_q.z(), relative_yaw;
    posegraph.updateKeyFrameLoop(index, loop_info);
  }

  void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
    // ROS_INFO("vio_callback!");
    Vector3d vio_t(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio * vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;

    if (!VISUALIZE_IMU_FORWARD) {
      cameraposevisual->reset();
      cameraposevisual->add_pose(vio_t_cam, vio_q_cam);
      cameraposevisual->publish_by(pub_camera_pose_visual, pose_msg->header);
    }

    odometry_buf.push(vio_t_cam);
    if (odometry_buf.size() > 10) {
      odometry_buf.pop();
    }

    visualization_msgs::Marker key_odometrys;
    key_odometrys.header = pose_msg->header;
    key_odometrys.header.frame_id = "map";
    key_odometrys.ns = "key_odometrys";
    key_odometrys.type = visualization_msgs::Marker::SPHERE_LIST;
    key_odometrys.action = visualization_msgs::Marker::ADD;
    key_odometrys.pose.orientation.w = 1.0;
    key_odometrys.lifetime = ros::Duration();

    // static int key_odometrys_id = 0;
    key_odometrys.id = 0; // key_odometrys_id++;
    key_odometrys.scale.x = 0.1;
    key_odometrys.scale.y = 0.1;
    key_odometrys.scale.z = 0.1;
    key_odometrys.color.r = 1.0;
    key_odometrys.color.a = 1.0;

    for (unsigned int i = 0; i < odometry_buf.size(); i++) {
      geometry_msgs::Point pose_marker;
      Vector3d vio_t;
      vio_t = odometry_buf.front();
      odometry_buf.pop();
      pose_marker.x = vio_t.x();
      pose_marker.y = vio_t.y();
      pose_marker.z = vio_t.z();
      key_odometrys.points.push_back(pose_marker);
      odometry_buf.push(vio_t);
    }
    pub_key_odometrys.publish(key_odometrys);

    // not used
    if (!LOOP_CLOSURE) {
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header = pose_msg->header;
      pose_stamped.header.frame_id = "map";
      pose_stamped.pose.position.x = vio_t.x();
      pose_stamped.pose.position.y = vio_t.y();
      pose_stamped.pose.position.z = vio_t.z();
      no_loop_path.header = pose_msg->header;
      no_loop_path.header.frame_id = "map";
      no_loop_path.poses.push_back(pose_stamped);
      pub_vio_path.publish(no_loop_path);
    }
  }

  void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
    m_process.lock();
    tic =
        Vector3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                 pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z)
              .toRotationMatrix();
    m_process.unlock();
  }

  void process() {
    if (!LOOP_CLOSURE)
      return;
    while (true) {
      sensor_msgs::ImageConstPtr image_msg = NULL;
      sensor_msgs::PointCloudConstPtr point_msg = NULL;
      nav_msgs::Odometry::ConstPtr pose_msg = NULL;
      // find out the messages with same time stamp
      m_buf.lock();
      // get image_msg, pose_msg and point_msg
      if (!image_buf.empty() && !point_buf.empty() && !pose_buf.empty()) {
        if (image_buf.front()->header.stamp.toSec() >
            pose_buf.front()->header.stamp.toSec()) {
          pose_buf.pop();
          printf("throw pose at beginning\n");
        } else if (image_buf.front()->header.stamp.toSec() >
                   point_buf.front()->header.stamp.toSec()) {
          point_buf.pop();
          printf("throw point at beginning\n");
        } else if (image_buf.back()->header.stamp.toSec() >=
                       pose_buf.front()->header.stamp.toSec() &&
                   point_buf.back()->header.stamp.toSec() >=
                       pose_buf.front()->header.stamp.toSec()) {
          pose_msg = pose_buf.front();
          pose_buf.pop();
          while (!pose_buf.empty())
            pose_buf.pop();
          while (image_buf.front()->header.stamp.toSec() <
                 pose_msg->header.stamp.toSec()) {
            image_buf.pop();
          }
          image_msg = image_buf.front();
          image_buf.pop();

          while (point_buf.front()->header.stamp.toSec() <
                 pose_msg->header.stamp.toSec())
            point_buf.pop();
          point_msg = point_buf.front();
          point_buf.pop();
        }
      }
      m_buf.unlock();
      if (pose_msg != NULL) {
        // printf(" pose time %f \n", pose_msg->header.stamp.toSec());
        // printf(" point time %f \n", point_msg->header.stamp.toSec());
        // printf(" image time %f \n", image_msg->header.stamp.toSec());
        // skip fisrt few
        if (skip_first_cnt < SKIP_FIRST_CNT) {
          skip_first_cnt++;
          continue;
        }

        if (skip_cnt < SKIP_CNT) {
          skip_cnt++;
          continue;
        } else {
          skip_cnt = 0;
        }

        cv::Mat image =
            cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8)
                ->image;
        // build keyframe
        Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                              pose_msg->pose.pose.position.y,
                              pose_msg->pose.pose.position.z);
        Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                 pose_msg->pose.pose.orientation.x,
                                 pose_msg->pose.pose.orientation.y,
                                 pose_msg->pose.pose.orientation.z)
                         .toRotationMatrix();

        //将距上一关键帧距离（平移向量的模）超过SKIP_DIS的图像创建为关键帧
        if ((T - last_t).norm() > SKIP_DIS) {
          vector<cv::Point3f> point_3d;
          vector<cv::Point2f> point_2d_uv;
          vector<cv::Point2f> point_2d_normal;
          vector<double> point_id;

          for (unsigned int i = 0; i < point_msg->points.size(); i++) {
            cv::Point3f p_3d;
            p_3d.x = point_msg->points[i].x;
            p_3d.y = point_msg->points[i].y;
            p_3d.z = point_msg->points[i].z;
            point_3d.push_back(p_3d);

            cv::Point2f p_2d_uv, p_2d_normal;
            double p_id;
            p_2d_normal.x = point_msg->channels[i].values[0];
            p_2d_normal.y = point_msg->channels[i].values[1];
            p_2d_uv.x = point_msg->channels[i].values[2];
            p_2d_uv.y = point_msg->channels[i].values[3];
            p_id = point_msg->channels[i].values[4];
            point_2d_normal.push_back(p_2d_normal);
            point_2d_uv.push_back(p_2d_uv);
            point_id.push_back(p_id);

            // printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
          }

          // 通过frame_index标记对应帧
          KeyFrame *keyframe = new KeyFrame(
              pose_msg->header.stamp.toSec(), frame_index, T, R, image,
              point_3d, point_2d_uv, point_2d_normal, point_id, sequence);
          m_process.lock();
          start_flag = 1;
          //在posegraph中添加关键帧，flag_detect_loop=1回环检测
          posegraph.addKeyFrame(keyframe, 1);
          m_process.unlock();
          frame_index++;
          last_t = T;
        }
      }

      std::chrono::milliseconds dura(5);
      std::this_thread::sleep_for(dura);
    }
  }

  void command() {
    if (!LOOP_CLOSURE)
      return;
    while (1) {
      char c = getchar();
      if (c == 's') {
        m_process.lock();
        posegraph.savePoseGraph();
        m_process.unlock();
        printf("save pose graph finish\nyou can set 'load_previous_pose_graph' "
               "to 1 in the config file to reuse it next time\n");
        printf("program shutting down...\n");
        ros::shutdown();
      }
      if (c == 'n')
        new_sequence();

      std::chrono::milliseconds dura(5);
      std::this_thread::sleep_for(dura);
    }
  }
};
PLUGINLIB_EXPORT_CLASS(pose_graph_nodelet_ns::PoseGraphNodelet,
                       nodelet::Nodelet)
} // namespace pose_graph_nodelet_ns
