#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include <sstream>
#include <boost/filesystem.hpp>
#include "vanish.hpp"
#include "Config.hpp"
#include "base.hpp"
#include "detector.hpp"

#define PI 3.1415926
#define CENTER_RANGE 0.1

using namespace std;
using namespace cv;
using namespace boost::filesystem;

Point read_vp(string p) {
    ifstream infile(p);
    Point pt;
    infile >> pt.x >> pt.y;
    return pt;
}

struct Result{
    double time_per_frame;
    double detected_rate;
    double dist_average;
    double pose_accuracy;
    int smpl_cnts;
    void print(std::ostream& stream, const string split="\n") {
        stream << "time_per_frame=" << time_per_frame << " ms"<<split;
        stream << "detected_rate=" << detected_rate*100 << "%" << split;
        stream << "dist_average=" << dist_average<< " pixels "<< split;
        stream << "smpl_cnts=" << smpl_cnts<< split;
        stream << "pose_accuracy=" << pose_accuracy * 100 << "%"  << split;
    }
    Result operator ^ (const Result & other) const {
        Result merged;
        merged.smpl_cnts = smpl_cnts + other.smpl_cnts;
        merged.time_per_frame = (time_per_frame * smpl_cnts +\
                other.time_per_frame * other.smpl_cnts)/merged.smpl_cnts;
        merged.detected_rate = (detected_rate * smpl_cnts + \
                other.detected_rate * other.smpl_cnts) / merged.smpl_cnts;
        merged.pose_accuracy = (pose_accuracy * smpl_cnts + \
                other.pose_accuracy * other.smpl_cnts) / merged.smpl_cnts;
        merged.dist_average = (dist_average * smpl_cnts * detected_rate + \
                other.dist_average* other.smpl_cnts * other.detected_rate) / (merged.smpl_cnts * merged.detected_rate);
        return merged;
    }
};
std::ostream & operator<< (std::ostream & s, Result rst) {
    rst.print(s);
    return s;
}

enum CamPose{
    Left, Center, Right
};

CamPose check_pose(const Point & cur_pt, const Point & focus_pt) {
    static const int BIAS = configs["center_bias"];
    if(cur_pt.x < focus_pt.x-BIAS) {
        return CamPose::Left;
    }else if(cur_pt.x > focus_pt.x+BIAS) {
        return CamPose::Right;
    }else {
        return CamPose::Center;
    }
}

CamPose predict_pose(const Point & cur_pt, const Point & focus_pt,
        const vector<Vec2f> & l_lines, const vector<Vec2f> & r_lines) {
    if(cur_pt.x == 0 && cur_pt.y == 0) {
        if(l_lines.empty() && !r_lines.empty()) {
            return CamPose::Left;
        }else if(!l_lines.empty() && r_lines.empty()) {
            return CamPose::Right;
        }else {
            return CamPose::Center;
        }
    }
    return check_pose(cur_pt, focus_pt);
}

Result evaluate(const string src_dir, string dst_dir, const int start_id,
        const int last_id) {
    boost::filesystem::path src_path(src_dir);
    string smpl_name = src_path.filename().string();
    dst_dir = dst_dir + smpl_name + "/";
    static const int BIAS = configs["center_bias"];
    ImgLogger edge_log(dst_dir, string("edge"));
    ImgLogger line_rgb(dst_dir, string("line_rgb"));
    ImgLogger line_edge(dst_dir, string("line_edge"));
    ImgLogger rgb_log(dst_dir, string("rgb"));
    const pair<double, double> r_theta_ranges{10*PI/180, 60 * PI / 180};
    const pair<double, double> l_theta_ranges{110 * PI / 180, 170 * PI / 180};
    VPDetector vp_detector(l_theta_ranges, r_theta_ranges);

    int id = 0;
    double time_elaps = 0.;
    double cnt = 0;
    double average = 0.0;
    int missing_cnt = 0;
    double sum_dist = 0.0;
    const Point focus_pt = read_vp(src_dir+to_string(start_id)+".txt");
    int pose_accurates = 0;
    for(int i = start_id; i <= last_id; ++i) {
        Mat image, dst, cdst, edge;
        image = imread(src_dir+to_string(i)+".jpg");
        const Point real_vp = read_vp(src_dir+to_string(i)+".txt");
        if (image.empty()) {
            throw std::logic_error("img "+to_string(i)+" is broken");
        }

        rgb_log.save(image, id);
        //cur_vp = vanish_point_detection(image, cdst, edge, time_elaps);
        vector<Vec2f> l_lines;
        vector<Vec2f> r_lines;
        //Point cur_vp = vp_detector.detect_vp(image, cdst, edge, time_elaps, l_lines, r_lines);
		Point cur_vp = vanish_point_detection(image, cdst, edge, time_elaps, l_lines, r_lines);
        if(cur_vp.x == 0 && cur_vp.y == 0) {
            ++missing_cnt;
        }else {
            cout << "real:" << real_vp << ", detected:" << cur_vp << "\n";
            double dist = cv::norm(cur_vp - real_vp);
            sum_dist += dist;
        }

        if(check_pose(real_vp, focus_pt) == predict_pose(cur_vp, focus_pt, l_lines, r_lines)) {
            ++ pose_accurates;
        }

        average = average * (cnt/(cnt+1)) + time_elaps /(cnt+1);
        ++cnt;
        cout << "It took " << time_elaps << " ms." << endl;
        circle(image, real_vp, 3, Scalar(0, 0, 255));
        line(image, Point(focus_pt.x-BIAS, 0), Point(focus_pt.x-BIAS, image.rows), Scalar(100, 100, 100), 1, CV_AA);
        line(image, Point(focus_pt.x+BIAS, 0), Point(focus_pt.x+BIAS, image.rows), Scalar(100, 100, 100), 1, CV_AA);
        edge_log.save(edge, id);
        line_edge.save(cdst, id);
        line_rgb.save(image, id);
        ++id;
        //imshow("rgb", image);
        //waitKey(10);
    }

    Result rst;
    rst.smpl_cnts = (last_id - start_id+1);
    rst.pose_accuracy = pose_accurates/ double(rst.smpl_cnts);
    rst.time_per_frame = average;
    rst.detected_rate = double(last_id-start_id+1-missing_cnt)/rst.smpl_cnts;
    rst.dist_average = sum_dist/rst.smpl_cnts;
    return rst;
}

int main(int argc, const char ** argv)
{
    Point mark_point, cur_vp;
    Mat image;

    Mat dst, cdst, edge;
    //double start = double(getTickCount());

    if(argc != 2) {
        cout << "Error! \nusage example ./bin/exe ../param/a.txt\n";
    }
    configs.init(argv[1]);
    stringstream dst_dir_ss( (string(configs["result_dir"])) );
    stringstream src_dir_ss( (string(configs["sample_dir"])) );
    stringstream start_id_ss( (string(configs["start_id"])) );
    stringstream last_id_ss( (string(configs["last_id"])) );
    

    string dst_dir;
    string src_dir;
    int start_id;
    int last_id;
    vector<Result> rst_vec;
    vector<string> src_vec;
    while(dst_dir_ss) {
        dst_dir_ss >> dst_dir;
        src_dir_ss >> src_dir;
        start_id_ss >> start_id;
        last_id_ss >> last_id;
        src_vec.push_back(src_dir);
        Result rst = evaluate(src_dir, dst_dir, start_id, last_id);
        rst_vec.push_back(rst);
    }

    for(size_t i = 0; i < rst_vec.size(); ++i) {
        cout << src_vec[i] << '\n';
        cout << rst_vec[i] << "\n----------\n";
    }

    Result total_rst = rst_vec[0];
    for(size_t i = 1; i < rst_vec.size(); ++i) {
        total_rst = total_rst ^ rst_vec[i];
    }
    cout << "total_rst=" <<total_rst;

    return 0;
}
