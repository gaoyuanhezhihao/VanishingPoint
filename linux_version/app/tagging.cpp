#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <numeric>
#include "vanish.hpp"
#include "Config.hpp"
#include "base.hpp"
using namespace std;
using namespace cv;

array<cv::Point, 2> click_pts;
int click_cnt = 0;
int hough_thres = 150;
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN ) {
        if(click_cnt < 2 ) {
            click_pts[click_cnt++] = Point(x, y);
        }else {
            cout << "WARNING no more clicks!" << endl;
        }
    }
}

char show_waitKey(Mat rgb, const vector<Vec2f> & lines, const int select=-1)  {
    draw_lines(rgb, lines, Scalar(0, 100, 100));
    if(-1 != select) {
        draw_lines(rgb, vector<Vec2f> {lines[select]}, GREEN);
    }
    imshow("window", rgb);
    return waitKey(0);
}

char show_waitClick(Mat rgb, const vector<Vec2f> & lines) {
    draw_lines(rgb, lines, Scalar(0, 100, 100));
    click_cnt = 0;
    char order = ' ';
    while(click_cnt < 2 && order != 'i' && order != 'd') {
        imshow("window", rgb);
        order = waitKey(10);
        //cout << "order =" << order <<"\n";
    }
    return order;
}

auto cmpor = [](const Vec2f & l1, const Vec2f & l2) {
    const int dist1 = dist_pt2line(l1, click_pts[0]) + \
                      dist_pt2line(l1, click_pts[1]);
    const int dist2 = dist_pt2line(l2, click_pts[0]) + \
                      dist_pt2line(l2, click_pts[1]);
    return dist1 < dist2;
};
//bool select_one_line(Mat rgb, vector<Vec2f> lines, Vec2f & the_line) {
    //char order=' ';
    //size_t si;
    //show_waitClick(rgb, lines);
    //std::sort(lines.begin(), lines.end(), cmpor);
    //for(si = 0; si < lines.size(); ++si) {
        //order = show_waitKey(rgb, lines, si);
        //if('y' == order) {
            //the_line = lines[si];
            //return true;
        //}
        ////if(order == 'i') {
            ////hough_thres += 10;
            ////break;
        ////}else if(order == 'd') {
            ////hough_thres -= 10;
            ////break;
        ////}else if(order == 'y' || order == 'r'){
            ////break;
        ////}
    //}
    //return false;
//}
bool select_one_line(Mat rgb, vector<Vec2f> lines, Vec2f & the_line) {
    char order=' ';
    size_t si;
    order = show_waitClick(rgb, lines);
    if(order == 'i') {
        hough_thres += 5;
        cout << "increase hough thres to " << hough_thres <<'\n';
        return false;
    }else if(order == 'd') {
        hough_thres -= 5;
        cout << "decreased hough thres to" << hough_thres <<'\n';
        return false;
    }
    std::sort(lines.begin(), lines.end(), cmpor);
    for(si = 0; si < lines.size(); ++si) {
        order = show_waitKey(rgb, lines, si);
        if(order == 'i') {
            hough_thres += 5;
            cout << "increase hough thres to " << hough_thres <<'\n';
            break;
        }else if(order == 'd') {
            hough_thres -= 5;
            cout << "decreased hough thres to" << hough_thres <<'\n';
            break;
        }else if(order == 'y' || order == 'r'){
            break;
        }
    }
    if(order == 'y') {
        the_line = lines[si];
        return true;
    }
    return false;
}
void adjust_hough(Mat rgb, const Mat & edge, vector<Vec2f> & lines) {
    char order = ' ';
    while(order != 'y') {
        lines.clear();
        HoughLines(edge, lines, 1, CV_PI / 180, hough_thres, 0, 0);
        order = show_waitKey(rgb.clone(), lines);
        if('i' == order ) {
            hough_thres += 10;
            cout << "increase hough thres to " << hough_thres <<'\n';
        }
        if('d' == order) {
            hough_thres -= 10;
            cout << "decreased hough thres to" << hough_thres <<'\n';
        }
    }
}

bool show_vp(Mat rgb, const Vec2f & l_line, const Vec2f& r_line, Point vp) {
    draw_lines(rgb, vector<Vec2f>{l_line, r_line});
    circle(rgb, vp, 5, Scalar(0, 150, 150));
    imshow("result", rgb);
    return 'y' == waitKey(10);
}
Point process(const Mat rgb) {
    Mat edge;
    vector<Vec2f> lines;
    Canny(rgb, edge, 30, 70, 3);
    vector<int> ids(lines.size());
    std::iota(ids.begin(), ids.end(), 0);
    //cout << "adjust hough thres\n";
    //adjust_hough(rgb.clone(), edge, lines);

    Vec2f l_line;

    do{
        lines.clear();
        HoughLines(edge, lines, 1, CV_PI / 180, hough_thres, 0, 0);
        cout << "choose left line\n";
    } while(!select_one_line(rgb.clone(), lines, l_line));
    Vec2f r_line;
    do{
        lines.clear();
        HoughLines(edge, lines, 1, CV_PI / 180, hough_thres, 0, 0);
        cout << "choose right line\n";
    }while(!select_one_line(rgb.clone(), lines, r_line));

    Point vp = intersect(l_line, r_line);
    show_vp(rgb.clone(), l_line, r_line, vp);
    return vp;
}

int main(int argc, const char ** argv) {
    if(argc != 2) {
        cout << "Error! \nusage example ./bin/vanish_pt ../param/configs\n";
    }
    configs.init(argv[1]);
    const string dst_dir = configs["result_dir"];
    VideoCapture cap(string(configs["video_path"]).c_str());
    if (!cap.isOpened())
    {
        cout << "Capture could not be opened successfully" << endl;
        return -1;
    }
    namedWindow("window", 1);
    setMouseCallback("window", CallBackFunc, NULL);
    int id = configs["start_id"];
    for(int i = 0; i < id; ++i) {
        Mat trash;
        cap >> trash;
    }

    while(cap.isOpened()){
        cout << id << "\n";
        Mat rgb;
        cap >> rgb;
        Point vp = process(rgb);
        imwrite(dst_dir+to_string(id)+".jpg", rgb);
        ofstream of;
        of.open(dst_dir+to_string(id)+".txt");
        of << vp.x << " " << vp.y << "\n";
        of.close();
        ++id;
    }
    //HoughLines(edge, lines, 1, CV_PI / 180, 150, 0, 0);
}
