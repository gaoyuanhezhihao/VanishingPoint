#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include "vanish.hpp"
#include "Config.hpp"
#include "base.hpp"
#include "detector.hpp"

#define PI 3.1415926
#define CENTER_RANGE 0.1

using namespace std;
using namespace cv;

Point read_vp(string p) {
    ifstream infile(p);
    Point pt;
    infile >> pt.x >> pt.y;
    return pt;
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
    const string dst_dir = configs["result_dir"];
    const string src_dir = configs["sample_dir"];
    const int start_id = configs["start_id"];
    const int last_id = configs["last_id"];
    ImgLogger edge_log(dst_dir, string("edge"));
    ImgLogger line_rgb(dst_dir, string("line_rgb"));
    ImgLogger line_edge(dst_dir, string("line_edge"));
    ImgLogger rgb_log(dst_dir, string("rgb"));
    char order = 0;

    int id = 0;
    double time_elaps = 0.;
    double cnt = 0;
    double average = 0.0;
    int missing_cnt = 0;
    double sum_dist = 0.0;
    const pair<double, double> r_theta_ranges{10*PI/180, 60 * PI / 180};
    const pair<double, double> l_theta_ranges{110 * PI / 180, 170 * PI / 180};
    VPDetector vp_detector(l_theta_ranges, r_theta_ranges);
    for(int i = start_id; i <= last_id; ++i) {
        Mat image, dst, cdst, edge;
        image = imread(src_dir+to_string(i)+".jpg");
        const Point real_vp = read_vp(src_dir+to_string(i)+".txt");
        if (image.empty()) {
            throw std::logic_error("img "+to_string(i)+" is broken");
        }
        else {
            rgb_log.save(image, id);
            //cur_vp = vanish_point_detection(image, cdst, edge, time_elaps);
            cur_vp = vp_detector.detect_vp(image, cdst, edge, time_elaps);
            if(cur_vp.x == 0 && cur_vp.y == 0) {
                ++missing_cnt;
            }else {
                cout << "real:" << real_vp << ", detected:" << cur_vp << "\n";
                double dist = cv::norm(cur_vp - real_vp);
                sum_dist += dist;
            }

            average = average * (cnt/(cnt+1)) + time_elaps /(cnt+1);
            ++cnt;
            cout << "It took " << time_elaps << " ms." << endl;
            edge_log.save(edge, id);
            line_edge.save(cdst, id);
            line_rgb.save(image, id);
            ++id;
            imshow("rgb", image);
            waitKey(10);
        }
    }
    //cout << "Press any key to exit:" << endl;
    //std::cin >> order;

    cout << "average time used for one frame:" << average << " ms\n";
    cout << double(last_id-start_id+1-missing_cnt)/(last_id-start_id+1) << "% detected\n";
    cout << "average distance = " << sum_dist/(last_id - start_id+1) << "\n";
    return 0;
}
