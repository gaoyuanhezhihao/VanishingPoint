#ifndef MAIN_FUNC_H
#define MAIN_FUNC_H
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;
bool hough_line_detect(Mat & image, Mat & cdst, vector<Vec2f> & left_lines, vector<Vec2f> & right_lines);
char vanish_point_detection(Mat & image, Mat & cdst);
bool draw_line(Mat & image, vector<Vec2f> & vec_lines, Scalar color);
#endif MAIN_FUNC_H