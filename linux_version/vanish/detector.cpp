#include <algorithm>
#include <numeric>
#include "detector.hpp"
#include "Config.hpp"
#include "range_hough.hpp"
#include "vanish.hpp"
#include "base.hpp"

using namespace std;
using namespace cv;

const int RHO = 0;
const int THETA = 1;

//class ThetaPredictor{
    //virtual pair<double, double> _predict_theta(const vector<Vec2f> & lines, const vector<int> & votes) {
        //static const double THETA_WIDTH = double(configs["theta_width_in_degree"])*CV_PI/180;
        //assert(lines.size() == votes.size());
        //assert(lines.size() > 0);
        //const double vote_sum = std::accumulate(votes.cbegin(), votes.cbegin(),0);
        //double theta_average = .0;
        //for(size_t i = 0; i < lines.size(); ++i) {
            //theta_average += lines[i][THETA] * votes[i]/ vote_sum;
        //}
        //pair<double, double> theta_rg;
        //theta_rg.first = CV_PI*max(0.0, theta_average - THETA_WIDTH)/180;
        //theta_rg.second = CV_PI*min(180.0, theta_average+THETA_WIDTH)/180;
        //return true;
    //}

//};

VanishLineDetector::VanishLineDetector(const std::pair<double, double> init_theta_rg):_theta_rg{init_theta_rg} {
    static const int init_hough_thres = configs["init_hough_thres"];
    _hough_thres = init_hough_thres;
}

void VanishLineDetector::_filter_lines(vector<Vec2f> & lines, vector<int> & votes) {
    static const size_t max_line_num = int(configs["max_number_of_lines"]);
    if(max_line_num >= lines.size()) {
        return;
    }
    auto cmpor = [&votes](const int i1, const int i2) {
        return votes[i1] > votes[i2];
    };
    vector<int> ids(lines.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), cmpor);

    vector<Vec2f> new_lines(max_line_num);
    vector<int> new_votes(max_line_num);
    for(size_t i = 0; i < max_line_num; ++i) {
        new_lines[i] = lines[ids[i]];
        new_votes[i] =votes[ids[i]];
    }
    swap(lines, new_lines);
    swap(votes, new_votes);
    return ;
}
bool VanishLineDetector::_predict_theta(const vector<Vec2f> & lines, const vector<int> & votes) {
    static const double THETA_WIDTH = double(configs["theta_width_in_degree"])*CV_PI/180;
    assert(lines.size() == votes.size());
    assert(lines.size() > 0);
    const double vote_sum = std::accumulate(votes.cbegin(), votes.cend(),0);
    assert(vote_sum > 0.);
    //cout << "vote_sum = " << vote_sum << endl;
    double theta_average = 0.0;
    for(size_t i = 0; i < lines.size(); ++i) {
        theta_average += lines[i][THETA] * votes[i]/ vote_sum;
    }
    //cout << "theta_average=" <<  theta_average *180/CV_PI << endl;
    //_theta_records.push_back(theta_average);
    //if(_theta_records.size() >=2) {
        //double chg = _theta_records.back() - _theta_records[_theta_records.size()-2];
        //_theta_predicted = theta_average + chg;
    //}
    
    _theta_rg.first = max(0.0, theta_average-THETA_WIDTH);
    _theta_rg.second = min(180.0, theta_average+THETA_WIDTH);
    //cout << "_predict_theta: theta range=(" << _theta_rg.first << "," << _theta_rg.second << "\n";
    return true;
}

bool VanishLineDetector::_predict_hough_thres(const vector<Vec2f> & lines, const vector<int> & votes) {
    static const int hough_width = configs["hough_thres_width"];
    _hough_thres = *std::min_element(votes.cbegin(), votes.cend());
    _hough_thres -= hough_width;
    //cout << "predicted hough thres=" << _hough_thres << "\n";
    return true;
}

bool VanishLineDetector::detect(const cv::Mat & edge, vector<Vec2f> & lines) {
    /* detect line */
    static const int max_retry = configs["max_retry"];
    static const size_t max_line_num = int(configs["max_number_of_lines"]);
    static const int HOUGH_RETRY_DEC = configs["hough_retry_decrease"];

    vector<int> hough_votes;
    int try_cnt = 0;
    for(try_cnt = 0; try_cnt <= max_retry; ++try_cnt) {
        hough_votes.clear();
        traceable_range_hough(edge,
                vector<pair<double, double>>{_theta_rg},
                _hough_thres, lines, hough_votes);
        if(0 != lines.size()) {
            break;
        }
        _hough_thres -= HOUGH_RETRY_DEC;
    }
    if(try_cnt > max_retry)  {
        return false;
    }
    if(lines.size() > max_line_num) {
        _filter_lines(lines, hough_votes);
    }
    //for(size_t i = 0; i < lines.size(); ++i) {
        //cout << '(' << lines[i][THETA] << "," << hough_votes[i] << "\n";
    //}
    /* predict next frame */
    _predict_theta(lines, hough_votes);
    _predict_hough_thres(lines, hough_votes);
    return true;
}

VPDetector::VPDetector(const std::pair<double, double>& init_l_theta_rg, const std::pair<double, double> & init_r_theta_rg):l_line_detector(init_l_theta_rg),r_line_detector(init_r_theta_rg) {}

vector<Point> intersect(vector<Vec2f> & l_lines, vector<Vec2f> & r_lines) {
	vector<Point> Intersection;
    int x =0;
    int y = 0;
    for (size_t i = 0; i < l_lines.size(); ++i) {
        for (size_t j = 0; j < r_lines.size(); ++j) {
            float rho_l = l_lines[i][0], theta_l = l_lines[i][1];
            float rho_r = r_lines[j][0], theta_r = r_lines[j][1];
            double denom = (sin(theta_l)*cos(theta_r) - cos(theta_l)*sin(theta_r));
            x = (rho_r*sin(theta_l) - rho_l*sin(theta_r)) / denom;
            y = (rho_l*cos(theta_r) - rho_r*cos(theta_l)) / denom;
            Point pt(x, y);
            Intersection.push_back(pt);
        }
    }
    return Intersection;
}

Point VPDetector::detect_vp(Mat & image, Mat & cdst, Mat & edge, double & time_used, vector<Vec2f>&l_lines, vector<Vec2f> & r_lines) {
    Canny(image, edge, 30, 70, 3);
    //vector<Vec2f> l_lines;
    //vector<Vec2f> r_lines;
    const double start = double(getTickCount());
    l_line_detector.detect(edge, l_lines);
    r_line_detector.detect(edge, r_lines);
    time_used = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    vector<Point> intersects = intersect(l_lines, r_lines);

    /* log */
    cvtColor(edge, cdst, CV_GRAY2BGR);
    draw_line(cdst, l_lines, Scalar(0, 255, 0));
    draw_line(cdst, r_lines, Scalar(255, 0, 0));
    draw_line(image, l_lines, Scalar(0, 255, 0));
    draw_line(image, r_lines, Scalar(255, 0, 0));
    draw_points(cdst, intersects, Scalar(0, 150, 150));
    draw_points(image, intersects, Scalar(0, 150, 150));
    return get_vanish_point(intersects);
}

