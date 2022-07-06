#include "feature_tracker.h"
#include <cstddef>
#include <future>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    p_fast_feature_detector = cv::FastFeatureDetector::create();
    n_id                    = 0;
}

void FeatureTracker::initGridsDetector()
{
    pool = std::make_shared<ThreadPool>(NUM_THREADS);

    grids_detector_img = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(0));

    grid_height     = (int)(ROW / NUM_GRID_ROWS);
    grid_width      = (int)(COL / NUM_GRID_COLS);
    grid_res_height = ROW - (NUM_GRID_ROWS - 1) * grid_height;
    grid_res_width  = COL - (NUM_GRID_COLS - 1) * grid_width;

    if (grids_rect.empty())
    {
        for (int i = 0; i < NUM_GRID_ROWS; i++)
            for (int j = 0; j < NUM_GRID_COLS; j++)
            {
                cv::Rect rect;
                if (i == 0)
                {
                    if (j == 0)
                        rect = cv::Rect(0, 0, grid_width + 3, grid_height + 3);
                    else if (j > 0 && j < NUM_GRID_COLS - 1)
                        rect = cv::Rect(j * grid_width - 3, 0, grid_width + 6, grid_height + 3);
                    else
                        rect = cv::Rect(j * grid_width - 3, 0, grid_res_width + 3, grid_height + 3);
                }
                else if (i > 0 && i < NUM_GRID_ROWS - 1)
                {
                    if (j == 0)
                        rect = cv::Rect(0, i * grid_height - 3, grid_width + 3, grid_height + 6);
                    else if (j > 0 && j < NUM_GRID_COLS - 1)
                        rect = cv::Rect(j * grid_width - 3, i * grid_height - 3, grid_width + 6,
                                        grid_height + 6);
                    else
                        rect = cv::Rect(j * grid_width - 3, i * grid_height - 3, grid_res_width + 3,
                                        grid_height + 6);
                }
                else
                {
                    if (j == 0)
                        rect =
                            cv::Rect(0, i * grid_height - 3, grid_width + 3, grid_res_height + 3);
                    else if (j > 0 && j < NUM_GRID_COLS - 1)
                        rect = cv::Rect(j * grid_width - 3, i * grid_height - 3, grid_width + 6,
                                        grid_res_height + 3);
                    else
                        rect = cv::Rect(j * grid_width - 3, i * grid_height - 3, grid_res_width + 3,
                                        grid_res_height + 3);
                    // if (j < NUM_GRID_COLS - 1)
                    //   rect = cv::Rect(j * grid_width, i * grid_height, grid_width,
                    //                   grid_res_height);
                    // else //右下角这个图像块
                    //   rect = cv::Rect(j * grid_width, i * grid_height, grid_res_width,
                    //                   grid_res_height);
                }
                grids_rect.emplace_back(rect);
                grids_track_num.emplace_back(0);
                grids_texture_status.emplace_back(true);
            }
    }

    grids_threshold = (int)(MAX_CNT / grids_rect.size());
    ROS_ASSERT_MSG(grids_threshold > 0,
                   "Too many grids! \n'max_cnt'(%d) is supposed to be bigger than "
                   "'num_grid_rows(%d)' x 'num_grid_cols(%d)' = %d!\nPlease reduce the "
                   "'num_grid_rows' or 'num_grid_cols'!",
                   MAX_CNT, NUM_GRID_ROWS, NUM_GRID_COLS, NUM_GRID_ROWS * NUM_GRID_COLS);
}

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int       img_x       = cvRound(pt.x);
    int       img_y       = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y &&
           img_y < ROW - BORDER_SIZE;
}

std::vector<cv::KeyPoint> FeatureTracker::gridDetect(size_t grid_id)
{
    // TicToc t_temp_ceil_fast;
    std::vector<cv::KeyPoint> temp_keypts;
    p_fast_feature_detector->detect(forw_img(grids_rect[grid_id]), temp_keypts,
                                    mask(grids_rect[grid_id]));

    if (SHOW_TRACK)
    {
        forw_img(grids_rect[grid_id]).copyTo(grids_detector_img(grids_rect[grid_id]));
        cv::drawKeypoints(grids_detector_img(grids_rect[grid_id]), temp_keypts,
                          grids_detector_img(grids_rect[grid_id]), cv::Scalar::all(255),
                          cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }

    if (temp_keypts.empty())
    {
        grids_texture_status[grid_id] = false;
        return {};
    }
    else
    {
        size_t grid_num_to_add = grids_threshold - grids_track_num[grid_id] + 2;
        if (temp_keypts.size() <= grid_num_to_add)
        {
            for (auto &temp_keypt : temp_keypts)
            {
                temp_keypt.pt.x += grids_rect[grid_id].x;
                temp_keypt.pt.y += grids_rect[grid_id].y;
            }
            return temp_keypts;
        }
        std::vector<cv::KeyPoint> keypts_to_add(grid_num_to_add);
        int                       min_response_id = 0;
        for (size_t j = 0; j < temp_keypts.size(); ++j)
        {
            if (grid_num_to_add > 0)
            {
                temp_keypts[j].pt.x += grids_rect[grid_id].x;
                temp_keypts[j].pt.y += grids_rect[grid_id].y;
                keypts_to_add[j] = temp_keypts[j];
                --grid_num_to_add;
                if (temp_keypts[j].response < keypts_to_add[min_response_id].response)
                {
                    min_response_id = j;
                }
            }
            else if (temp_keypts[j].response > keypts_to_add[min_response_id].response)
            {
                temp_keypts[j].pt.x += grids_rect[grid_id].x;
                temp_keypts[j].pt.y += grids_rect[grid_id].y;
                keypts_to_add[min_response_id] = temp_keypts[j];
                for (size_t k = 0; k < keypts_to_add.size(); ++k)
                {
                    if (keypts_to_add[k].response < keypts_to_add[min_response_id].response)
                    {
                        min_response_id = k;
                    }
                }
            }
        }
        keypts_to_add.resize(keypts_to_add.size() - grid_num_to_add);
        return keypts_to_add;
    }
    // printf("detect grids_img feature costs: %fms\n",
    //        t_temp_ceil_fast.toc());
}

void FeatureTracker::setMask()
{
    if (FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.emplace_back(track_cnt[i], make_pair(forw_pts[i], ids[i]));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         { return a.first > b.first; });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
    for (auto &pt : unstable_pts)
    {
        cv::circle(mask, pt, MIN_DIST, 0, -1);
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::addPoints(vector<cv::KeyPoint> &Keypts)
{
    for (auto &Keypt : Keypts)
    {
        if (mask.at<uchar>(Keypt.pt) == 255)
        {
            forw_pts.push_back(Keypt.pt);
            ids.push_back(-1);
            track_cnt.push_back(1);
            // cl: prevent close feature selected
            cv::circle(mask, Keypt.pt, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints(int n_max_cnt, vector<cv::KeyPoint> &Keypts)
{
    if (Keypts.empty())
    {
        return;
    }

    sort(Keypts.begin(), Keypts.end(),
         [](const cv::KeyPoint &a, const cv::KeyPoint &b) { return a.response > b.response; });

    int n_add = 0;
    for (auto &Keypt : Keypts)
    {
        if (mask.at<uchar>(Keypt.pt) == 255)
        {
            forw_pts.push_back(Keypt.pt);
            ids.push_back(-1);
            track_cnt.push_back(1);
            cv::circle(mask, Keypt.pt, MIN_DIST, 0, -1);  // cl: prevent close feature selected
            n_add++;
            if (n_add == n_max_cnt)
            {
                break;
            }
        }
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time, const Matrix3d &_relative_R)
{
    cv::Mat img;
    TicToc  t_r;
    cur_time = _cur_time;
    // too dark or too bright: histogram
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc             t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        // curr_img<--->forw_img
        cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();
    unstable_pts.clear();

    if (!cur_pts.empty())
    {
        TicToc        t_o;
        vector<uchar> status;
        vector<float> err;

        if (USE_IMU)
        {
            predictPtsInNextFrame(_relative_R);
            forw_pts = predict_pts;
            cv::calcOpticalFlowPyrLK(
                cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 1,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);
        }
        else
        {
            cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err,
                                     cv::Size(21, 21), 3);
        }

        for (int i = 0; i < int(forw_pts.size()); i++)
        {
            if (!status[i] && inBorder(forw_pts[i]))
            {
                unstable_pts.push_back(forw_pts[i]);
            }
            else if (status[i] && !inBorder(forw_pts[i]))
            {
                status[i] = 0;
            }
        }

        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);

        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        //  光流准确率，时间输出
        static double OpticalFlow_time = 0;
        static int    of_cnt_frame     = 0;
        OpticalFlow_time += t_o.toc();
        of_cnt_frame++;
        ROS_DEBUG("average optical flow costs: %fms\n", OpticalFlow_time / (double)of_cnt_frame);
        // static double succ_num = 0;
        // static double total_num = 0;
        // for (size_t i = 0; i < status.size(); i++) {
        //   if (status[i])
        //     succ_num++;
        // }
        // total_num += status.size();
        // ROS_DEBUG("average success ratio: %f\n", succ_num / total_num);
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        // TicToc t_m;
        //对cur_pts和forw_pts做ransac剔除outlier.
        rejectWithF();
        setMask();
        // ROS_DEBUG("set mask costs %fms", t_m.toc());

        TicToc t_t;
        int    n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;

            // TicToc t_grid_detect;

            for (auto &grid_track_num : grids_track_num)
                grid_track_num = 0;

            for (auto &forw_pt : forw_pts)
            {
                int col_id = (int)forw_pt.x / grid_width;
                int row_id = (int)forw_pt.y / grid_height;
                if (col_id == NUM_GRID_COLS)
                    --col_id;
                if (row_id == NUM_GRID_ROWS)
                    --row_id;
                ++grids_track_num[col_id + NUM_GRID_COLS * row_id];
            }

            std::vector<int> grids_id;
            for (size_t i = 0; i < grids_rect.size(); ++i)
            {
                if (grids_track_num[i] < grids_threshold && grids_texture_status[i])
                {
                    grids_id.emplace_back(i);
                }
                else
                {
                    grids_texture_status[i] = true;
                }
            }

            std::vector<std::future<std::vector<cv::KeyPoint>>> grids_keypts;
            for (int &i : grids_id)
            {
                grids_keypts.emplace_back(
                    pool->enqueue([this](int grid_id) { return gridDetect(grid_id); }, i));
            }

            // TicToc t_a;
            for (auto &&grid_keypts_asyn : grids_keypts)
            {
                std::vector<cv::KeyPoint> &&grid_keypts = grid_keypts_asyn.get();
                addPoints(grid_keypts);
            }
            // std::vector<cv::KeyPoint> Keypts;
            // p_fast_feature_detector->detect(forw_img, Keypts, mask);
            // addPoints(n_max_cnt, Keypts);

            // ROS_DEBUG("selectFeature costs: %fms", t_a.toc());

            /*  static double grid_detect_fast_time = 0;
             grid_detect_fast_time += t_grid_detect.toc();
             static int cnt_grid_frame = 0;
             cnt_grid_frame++;
             printf("average grids_img detect feature costs: %fms\n",
                    grid_detect_fast_time / (double)cnt_grid_frame); */
        }

        static double detect_time      = 0;
        static int    detect_cnt_frame = 0;
        detect_time += t_t.toc();
        detect_cnt_frame++;
        ROS_DEBUG("average detect costs: %fms\n", detect_time / (double)detect_cnt_frame);
        // ROS_DEBUG("detect feature costs: %fms", t_t.toc());
    }
    prev_un_pts = cur_un_pts;
    cur_img     = forw_img;
    cur_pts     = forw_pts;

    //  去畸变，投影至归一化平面，计算特征点速度(pixel/s)
    undistortedPoints();
    prev_time = cur_time;
    ROS_DEBUG("Process Image costs: %fms", t_r.toc());
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        TicToc              t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x()     = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y()     = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x()      = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y()      = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(),
                  1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

Eigen::Vector3d FeatureTracker::get3dPt(const cv::Mat &depth, const cv::Point2f &pt)
{
    Eigen::Vector3d tmp_P;
    m_camera->liftProjective(Eigen::Vector2d(pt.x, pt.y), tmp_P);
    Eigen::Vector3d P =
        tmp_P.normalized() * (((int)depth.at<unsigned short>(round(pt.y), round(pt.x))) / 1000.0);

    return P;
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_DEBUG("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat                 undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.emplace_back(b.x() / b.z(), b.y() / b.z());
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 &&
            pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) =
                cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x,
            // pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    // cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        // https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/0d280936e441ebb782bf8855d86e13999a22da63/camera_model/src/camera_models/PinholeCamera.cc
        // brief Lifts a point from the image plane to its projective ray
        m_camera->liftProjective(a, b);
        // 特征点在相机坐标系的归一化坐标
        cur_un_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        // printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.emplace_back(v_x, v_y);
                }
                else
                    pts_velocity.emplace_back(0, 0);
            }
            else
            {
                pts_velocity.emplace_back(0, 0);
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.emplace_back(0, 0);
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

void FeatureTracker::predictPtsInNextFrame(const Matrix3d &_relative_R)
{
    predict_pts.resize(cur_pts.size());
    for (unsigned int i = 0; i < cur_pts.size(); ++i)
    {
        Eigen::Vector3d tmp_P;
        m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_P);
        Eigen::Vector3d predict_P = _relative_R * tmp_P;
        Eigen::Vector2d tmp_p;
        m_camera->spaceToPlane(predict_P, tmp_p);
        predict_pts[i].x = tmp_p.x();
        predict_pts[i].y = tmp_p.y();
    }
}