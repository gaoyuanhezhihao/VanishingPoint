#ifndef VANISH_HPP
#define VANISH_HPP
#include <vector>
//#include "opencv2/highgui/highgui.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
//bool hough_line_detect(cv::Mat & image, cv::Mat & cdst, std::vector<cv::Vec2f> & left_lines, std::vector<cv::Vec2f> & right_lines);
cv::Point vanish_point_detection(cv::Mat & image, cv::Mat & cdst, cv::Mat & edge, double & time_used);
bool draw_line(cv::Mat & image, std::vector<cv::Vec2f> & vec_lines, cv::Scalar color);
cv::Point get_vanish_point(std::vector<cv::Point> & points);
#endif //VANISH_HPP
