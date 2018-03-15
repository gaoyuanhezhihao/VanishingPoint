#ifndef DETECTOR_H
#define DETECTOR_H

#include <list>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
class VanishLineDetector{
    private:
        std::list<int>  _hough_votes;
        std::list<std::pair<double, double>> _theta_rg_lst;
        int _hough_thres;
    public:
        VanishLineDetector(const std::pair<double, double> init_theta_rg); 
        bool detect(const cv::Mat & cur_frame); 
};
#endif //DETECTOR_H
