#ifndef RANGE_HOUGH_H
#define RANGE_HOUGH_H


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include <iostream>
#include <vector>

bool range_hough(const cv::Mat & edge_im, const std::vector<std::pair<double, double>> & theta_ranges, const int threshold, std::vector<cv::Vec2f> & lines);
void HoughLinesStandard( const cv::Mat& img, float rho, float theta,
                    int threshold, std::vector<cv::Vec2f>& lines, int linesMax,
                    double min_theta, double max_theta );
#endif //RANGE_HOUGH_H
