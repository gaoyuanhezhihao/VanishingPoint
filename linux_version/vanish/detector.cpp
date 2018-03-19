#include <algorithm>
#include <numeric>
#include "detector.hpp"
#include "Config.hpp"
#include "range_hough.hpp"
#include "vanish.hpp"
#include "base.hpp"
#include <queue>

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

inline double solve_y(const int x, const double rho, const double cos_theta, const double sin_theta) {
    return (rho - x*cos_theta)/sin_theta;
}
inline double solve_x(const int y, const double rho, const double cos_theta, const double sin_theta) {
    return (rho - y*sin_theta)/cos_theta;
}
vector<Point> line_endPoint_in_img(const cv::Size & img_size, const Vec2f & line) {
    const int cols = img_size.width;
    const int rows = img_size.height;
    const double rho = line[RHO];
    const double theta = line[THETA];
    const double cos_theta = cos(theta);
    const double sin_theta = sin(theta);
    vector<Point> candi{{0, -1}, {cols-1, -1}, {-1, 0}, {-1, rows-1}}; 
    vector<Point> rst;
    for(Point & pt: candi) {
        if(pt.x == -1) {
            pt.x = solve_x(pt.y, rho, cos_theta, sin_theta);
            if(0 <= pt.x && pt.x < cols) {
                rst.push_back(pt);
            }
        }else {
            pt.y = solve_y(pt.x, rho, cos_theta, sin_theta);
            if(0 <= pt.y && pt.y < rows) {
                rst.push_back(pt);
            }
        }
        if(rst.size() == 2) {
            break;
        }
    }
    return rst;
}

double calc_line_edge_ratio(const Mat & edge, const Vec2f & line) {
    static const int K = 3;
    static const vector<Point> d{{0, 0}, {0, 1}, {0, -1}};
    //static const vector<Point> d{{0, 0}, {0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    //static const vector<Point> d{{0, 0}};
    vector<Point> endPts = line_endPoint_in_img(edge.size(), line);
    CV_Assert(2 == endPts.size());
    CV_Assert(edge.depth() == CV_8U);
    CV_Assert(edge.channels() == 1);
    cv::LineIterator itr(edge, endPts[0], endPts[1], 8);
    int line_edge_cnt = 0;
    const int len = itr.count;
    int sum = 0;
    std::priority_queue<int> que;
    const int rows = edge.rows;
    const int cols = edge.cols;
    for(int i = 0; i < len; ++i, ++itr) {
        int j = 0;
        for(j = 0; j < d.size(); ++j) {
            Point pos(itr.pos()+d[j]);
            if(pos.y < rows && pos.y >= 0 && pos.x < cols && pos.x >= 0 &&\
                    edge.at<uchar>(pos)> 0) {
                break;
            }
        }
        if(j < d.size()) {
            ++sum;
            ++line_edge_cnt;
        } else {
            if(line_edge_cnt > 0) {
                que.push(line_edge_cnt);
                line_edge_cnt = 0;
            }
        }
    }
    if(line_edge_cnt > 0) {
        que.push(line_edge_cnt);
    }
    int sum_topK = 0;
    for(int i = 0; i < K && !que.empty(); ++i) {
        sum_topK += que.top();
        que.pop();
    }
    //CV_Assert(sum == 0);
    cout << "sum =" << sum << " sum_topK=" << sum_topK  << " len=" << len<< "\n";

    return sum == 0 ? 0.0: double(sum_topK)/sum;
}

pair<double, double> RangePredictor::predict(const vector<double> & cur) {
    assert(cur.size() > 0); 
    double min_v = *std::min_element(cur.cbegin(), cur.cend());
    double max_v = *std::max_element(cur.cbegin(), cur.cend());
    //cout << "min_v = "<< min_v << "\n";
    //cout << "max_v = "<< max_v << "\n";
    _records.emplace_back(min_v, max_v);
    const int sz = _records.size();
    if(sz >= 3) {
        double dmin_prev = _records[sz-1].first - _records[sz-2].first;
        double dmin_pprev = _records[sz-2].first - _records[sz-3].first;
        double dmin = dmin_prev + (dmin_prev - dmin_pprev);

        double dmax_prev = _records[sz-1].second - _records[sz-2].second;
        double dmax_pprev = _records[sz-2].second - _records[sz-3].second;
        double dmax = dmax_prev + (dmax_prev - dmax_pprev);
        return {min_v+dmin, max_v+dmax};
    }

    if(sz == 2) {
        double dmin_prev = _records[sz-1].first - _records[sz-2].first;
        double dmin = dmin_prev;
        double dmax_prev = _records[sz-1].second - _records[sz-2].second;
        double dmax = dmax_prev;
        return {min_v+dmin, max_v+dmax};
    }

    //cout << "---" << endl;
    //cout << "min_v = "<< min_v << "\n";
    //cout << "max_v = "<< max_v << "\n";
    return {min_v, max_v};
}

VanishLineDetector::VanishLineDetector(const std::pair<double, double> init_theta_rg):_theta_rg{init_theta_rg} {
    static const int init_hough_thres = configs["init_hough_thres"];
    _hough_thres = init_hough_thres;
}

void VanishLineDetector::_filter_lines_by_edge(vector<Vec2f> & lines, vector<int> &votes, const Mat & edge) {
    /* calc line edge ratio of lines */
    vector<double>  ratios(lines.size());
    cout << "-----------\n";
    for(size_t i = 0; i < lines.size(); ++i) {
        ratios[i] = calc_line_edge_ratio(edge, lines[i]);
        cout << ratios[i] <<  "vote=" << votes[i] << '\n';
    }

    static const size_t max_line_num = int(configs["max_number_of_lines"]);
    if(max_line_num >= lines.size()) {
        return;
    }
    auto cmpor = [&ratios](const int i1, const int i2) {
        return ratios[i1] > ratios[i2];
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
bool VanishLineDetector::_dynamic_predict_theta(const vector<Vec2f> & lines, const vector<int> & votes) {
    static const double THETA_WIDTH = double(configs["theta_width_in_degree"])*CV_PI/180;
    vector<double> thetas(lines.size());
    for(size_t i = 0; i < lines.size(); ++i) {
        thetas[i] = lines[i][THETA];
        //cout << thetas[i] << " ";
    }
    _theta_rg = _theta_predictor.predict(thetas);
    _theta_rg.first = max(0.0, _theta_rg.first-THETA_WIDTH);
    _theta_rg.second = min(CV_PI, _theta_rg.second+THETA_WIDTH);
    return true;
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
    _theta_records.push_back(theta_average);
    _theta_predicted = theta_average;
    //if(_theta_records.size() >=2) {
    //double chg = _theta_records.back() - _theta_records[_theta_records.size()-2];
    //_theta_predicted += chg;
    //}
    //cout << "_theta_predicted=" << _theta_predicted;

    _theta_rg.first = max(0.0, _theta_predicted-THETA_WIDTH);
    _theta_rg.second = min(CV_PI, _theta_predicted+THETA_WIDTH);
    //_theta_rg.first = max(0.0, theta_average-THETA_WIDTH);
    //_theta_rg.second = min(180.0, theta_average+THETA_WIDTH);
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
        //_filter_lines(lines, hough_votes);
        _filter_lines_by_edge(lines, hough_votes, edge);
    }
    //for(size_t i = 0; i < lines.size(); ++i) {
    //cout << '(' << lines[i][THETA] << "," << hough_votes[i] << "\n";
    //}
    /* predict next frame */
    _predict_theta(lines, hough_votes);
    //_dynamic_predict_theta(lines, hough_votes);
    //cout << "theta_rg =" << " "   << _theta_rg.first << ","<< _theta_rg.second << endl;
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
    const double start = double(getTickCount());

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);
    Canny(gray, edge, 30, 70, 3);
    //Canny(image, edge, 30, 70, 3);
    //vector<Vec2f> l_lines;
    //vector<Vec2f> r_lines;
    l_line_detector.detect(edge, l_lines);
    r_line_detector.detect(edge, r_lines);
    vector<Point> intersects = intersect(l_lines, r_lines);
    Point vp = get_vanish_point(intersects);

    time_used = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    /* log */
    cvtColor(edge, cdst, CV_GRAY2BGR);
    draw_line(cdst, l_lines, Scalar(0, 255, 0));
    draw_line(cdst, r_lines, Scalar(255, 0, 0));
    draw_line(image, l_lines, Scalar(0, 255, 0));
    draw_line(image, r_lines, Scalar(255, 0, 0));
    draw_points(cdst, intersects, Scalar(0, 150, 150));
    draw_points(image, intersects, Scalar(0, 150, 150));
    return vp;
}

