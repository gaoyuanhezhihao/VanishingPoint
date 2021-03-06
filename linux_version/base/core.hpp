#ifndef CORE_H
#define CORE_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Config.hpp"
//#include "Pose.hpp"

class NewMatch;
class NewFrame;


class NewFrame{
    private:
        int _id;
        cv::Mat _rgb;
        cv::Mat _gray;
        cv::Mat _edge;
        std::vector<cv::Point2f> _KeyPts;
        std::vector<cv::Vec2f> _lines;
        std::vector<std::vector<int>> _Line_endPt_id_map;
        cv::Mat _R_global;
        cv::Mat _t_global;
        std::vector<cv::Point2f> _global_pts;
        std::vector<NewFrame*> _pMatchFrames;
        std::vector<int> _match_keyPt_ids;
        double _theta;
        double _t[2];
        void add_restrict(vector<array<double, 2>> & pts_cur_local, const cv::Point2f & pt_local, NewFrame* pF, const int id, vector<array<double,2>> & pts_global);
    public:
        void set_as_oring();
        bool neib_BA();
        void add_match_pts(int this_id, int other, NewFrame* pF);

        void merge_tracked_lines(const std::vector<cv::Vec2f> & tracked_lines); 
        NewFrame(int id):_id(id), _R_global(cv::Mat::eye(2, 2, CV_64F)), _t_global(cv::Mat::zeros(2, 1, CV_64F)), _theta(0.), _t{0.,0.} {}
        double get_theta() {return _theta;}
        pair<double, double> get_coordinates() {return {_t[0], _t[1]};}

        //void set_lines(std::shared_ptr<std::vector<cv::Vec2f>> pl) {
        //assert(nullptr == _pLines);
        //_pLines = pl;
        //}

        bool calc_keyPts(); 
        bool detect_lines(const vector<pair<double, double>> & theta_rgs);
        bool detect_lines();

        //void set_key_pts(std::shared_ptr<std::vector<cv::Point2f>> pKeyPts,
        //std::shared_ptr<std::vector<std::vector<int>>> pLine_endPt_id_map) {
        //assert(nullptr == _pKeyPts);
        //assert(nullptr == _pLine_endPt_id_map);
        //_pKeyPts = pKeyPts;
        //_pLine_endPt_id_map = pLine_endPt_id_map;
        //}
        const std::vector<cv::Point2f> & keyPts()const {
            return _KeyPts;
        }
        //const std::vector<cv::Point2f> & keyPts()const {
        //return *_pKeyPts;
        //}
        const cv::Mat & rgb()const {
            return _rgb;
        }

        const cv::Mat & gray() const {
            return _gray;
        }
        const std::vector<std::vector<int>> line_endPt_id_map() const {
            return _Line_endPt_id_map;
        }
        //const std::vector<std::vector<int>> line_endPt_id_map() const {
        //return *_pLine_endPt_id_map;
        //}
        const cv::Mat & edge()const {
            return _edge;
        }
        const std::vector<cv::Vec2f> lines()const {
            //return *_pLines;
            return _lines;
        }
        int get_id() const{
            return _id;
        }

        void read_frame();

};

class NewMatch{
    private:
        NewFrame * pf1;
        NewFrame * pf2;
        std::vector<std::pair<int, int>> ids;
        cv::Mat R;
        cv::Mat t;
        double _dx=.0;
        double _dy=.0;
        double _theta=.0;
        //Pose2D _cam_mot;
        //Pose2D _car_mot;
    public:
        NewMatch(NewFrame* pframe1, NewFrame * pframe2):pf1(pframe1), pf2(pframe2) {

        }

        void add(int id1, int id2) ;
        cv::Mat draw() const; 
        bool calc_cam_motion();
        bool calc_car_motion();
        bool ceres_solve_cam_motion();
        double get_dx()const{return _dx;}
        double get_dy()const{return _dy;}
        double get_theta()const {return _theta;}
        int match_num()const {return ids.size();}
};

cv::Scalar rand_color(); 

cv::Mat draw_frame(const NewFrame & f);
#endif //CORE_H
