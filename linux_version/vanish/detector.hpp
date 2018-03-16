#ifndef DETECTOR_H
#define DETECTOR_H

#include <list>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
class VanishLineDetector{
    private:
        list<int>  _hough_votes;
        vector<double> _theta_records;
        double _theta_predicted;
        pair<double, double> _theta_rg;
        int _hough_thres;
        bool _predict_theta(const vector<Vec2f> & lines,
                const vector<int> & votes);
        bool _predict_hough_thres(const vector<Vec2f>&lines,
                const vector<int> & votes);
        void _filter_lines(vector<Vec2f> & lines, vector<int> & votes);
    public:
        VanishLineDetector(const pair<double, double> init_theta_rg); 
        bool detect(const Mat & cur_frame, vector<Vec2f> & lines); 
};

class VPDetector{
    private:
        VanishLineDetector l_line_detector;
        VanishLineDetector r_line_detector;
    public:
        VPDetector(const pair<double, double>& init_l_theta_rg, const pair<double, double> & init_r_theta_rg);
        Point detect_vp(Mat & image, Mat & cdst, Mat & edge, double & time_used, vector<Vec2f>&l_lines, vector<Vec2f> & r_lines);
};
#endif //DETECTOR_H
