#include <algorithm>
#include "detector.hpp"
#include "Config.hpp"
#include "range_hough.hpp"

using namespace std;
using namespace cv;

VanishLineDetector::VanishLineDetector(const std::pair<double, double> init_theta_rg):_theta_rg_lst{init_theta_rg} {
    static const int init_hough_thres = configs["init_hough_thres"];
    _hough_thres = init_hough_thres;
}

static void filter_lines(const vector<Vec2f> & lines, const vector<int> & votes) {
    auto cmpor = [&votes](const int i1, const int i2) {
        return votes[i1] > votes[i2];
    };

}

bool VanishLineDetector::detect(const cv::Mat & cur_frame, Mat & edge, vector<Vec2f> & lines) {
    /* detect line */
    static const int max_retry = configs["max_retry"];
    static const int max_line_num = configs["max_number_of_lines"];
    Canny(cur_frame, edge, 30, 70, 3);
    vector<pair<double, double>> theta_rg{_theta_rg_lst.back()};

    vector<int> hough_votes;
    int try_cnt = 0;
    for(try_cnt = 0; try_cnt <= max_retry; ++try_cnt) {
        hough_votes.clear();
        traceable_range_hough(edge, _theta_rg_lst.front(), _hough_thres, lines, hough_votes);
        if(0 != lines.size()) {
            break;
        }
    }
    if(try_cnt > max_retry)  {
        return false;
    }
    if(lines.size() > max_line_num) {

    }
}

