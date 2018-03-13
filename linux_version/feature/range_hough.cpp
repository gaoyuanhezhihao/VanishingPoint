#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <list>
#include <cstring>
#include <cstdlib>
#include "Config.hpp"

using namespace cv;
using namespace std;

struct hough_cmp_gt
{
    hough_cmp_gt(const int* _aux) : aux(_aux) {}
    bool operator()(int l1, int l2) const
    {
        return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
    }
    const int* aux;
};
struct LinePolar
{
    float rho;
    float angle;
};
struct hough_index
{
    hough_index() : value(0), rho(0.f), theta(0.f) {}
    hough_index(int _val, float _rho, float _theta)
    : value(_val), rho(_rho), theta(_theta) {}

    int value;
    float rho, theta;
};
void HoughLinesStandard( const Mat& img, float rho, float theta,
                    int threshold, std::vector<Vec2f>& lines, int linesMax,
                    double min_theta, double max_theta )
{
    int i, j;
    float irho = 1 / rho;

    CV_Assert( img.type() == CV_8UC1 );

    const uchar* image = img.ptr();
    int step = (int)img.step;
    int width = img.cols;
    int height = img.rows;

    if (max_theta < min_theta ) {
        CV_Error( CV_StsBadArg, "max_theta must be greater than min_theta" );
    }
    int numangle = cvRound((max_theta - min_theta) / theta);
    int numrho = cvRound(((width + height) * 2 + 1) / rho);


    AutoBuffer<int> _accum((numangle+2) * (numrho+2));
    std::vector<int> _sort_buf;
    AutoBuffer<float> _tabSin(numangle);
    AutoBuffer<float> _tabCos(numangle);
    int *accum = _accum;
    float *tabSin = _tabSin, *tabCos = _tabCos;

    memset( accum, 0, sizeof(accum[0]) * (numangle+2) * (numrho+2) );

    float ang = static_cast<float>(min_theta);
    for(int n = 0; n < numangle; ang += theta, n++ )
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }

    // stage 1. fill accumulator
    for( i = 0; i < height; i++ )
        for( j = 0; j < width; j++ )
        {
            if( image[i * step + j] != 0 )
                for(int n = 0; n < numangle; n++ )
                {
                    int r = cvRound( j * tabCos[n] + i * tabSin[n] );
                    r += (numrho - 1) / 2;
                    accum[(n+1) * (numrho+2) + r+1]++;
                }
        }

    // stage 2. find local maximums
    for(int r = 0; r < numrho; r++ )
        for(int n = 0; n < numangle; n++ )
        {
            int base = (n+1) * (numrho+2) + r+1;
            if( accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2] )
                _sort_buf.push_back(base);
        }

    // stage 3. sort the detected lines by accumulator value
    std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));

    // stage 4. store the first min(total,linesMax) lines to the output buffer
    linesMax = std::min(linesMax, (int)_sort_buf.size());
    double scale = 1./(numrho+2);
    for( i = 0; i < linesMax; i++ )
    {
        LinePolar line;
        int idx = _sort_buf[i];
        int n = cvFloor(idx*scale) - 1;
        int r = idx - (n+1)*(numrho+2) - 1;
        line.rho = (r - (numrho - 1)*0.5f) * rho;
        line.angle = static_cast<float>(min_theta) + n * theta;
        lines.push_back(Vec2f(line.rho, line.angle));
    }
}

bool range_hough(const cv::Mat & edge_im, const vector<pair<double, double>> & theta_ranges, const int threshold, vector<Vec2f> & lines) {
    static const double theta_resolution = double(configs["theta_resolution"]) * CV_PI / 180;
    static const double rho_resolution = configs["rho_resolution"];

    //SHOW(theta_resolution);
    //SHOW(rho_resolution);
    //const string dst_dir = configs["result_dir"];
    //imwrite(dst_dir+"range_hough_debug.jpg", edge_im);
    const int width = edge_im.cols;
    const int height = edge_im.rows;
    //CvMat c_image = edge_im;
    const uchar * image = edge_im.ptr();
    //image = c_image.data.ptr;

    vector<double> theta_vec;
    for(const pair<double, double> & rg: theta_ranges)  {
        double t = rg.first;
        while(t < rg.second) {
            theta_vec.push_back(t);
            t += theta_resolution;
        }
    }

    int numrho = cvRound(((width + height) * 2 + 1) / rho_resolution);
    if(theta_vec.empty()) {
        return false;
    }

    const float irho_rsv = 1/rho_resolution;
    int numangle = theta_vec.size();
    float * tabSin = new float[numangle];
    float * tabCos = new float[numangle];
    for(int i = 0; i < numangle; ++i) {
        double theta = theta_vec[i];
        tabSin[i] = (float) (sin(theta)*irho_rsv);
        tabCos[i] = (float) (cos(theta)*irho_rsv);
    }

    int * accum = new int[(numangle+2)*(numrho+2)];
    memset(accum, 0, sizeof(accum[0]) * (numangle+2)*(numrho+2));
    //vector<int*> cnt_ptr_vec;
    //for(int i = 0; i < numangle; ++i) {
        //cnt_ptr_vec[i] = accum + numrho * (i+1);
    //}

    // stage 1. fill accumulator.
    const int step = edge_im.step;
    const int zero_rho_idx = (numrho-1)/2;
    for(int r = 0; r < height; ++r) {
        for(int c = 0; c < width; ++c) {
            if(image[r * step + c] != 0) {
                for(int i = 0; i < numangle; ++i) {
                    int rho_id = cvRound(c * tabCos[i] + r * tabSin[i]);
                    rho_id += zero_rho_idx;
                    ++accum[(i+1)* (numrho+2)+ rho_id+1];
                }
            }
        }
    }
    
    //vector<int> base_vec;
    // stage 2. find local maximums
    //float max_theta = 0.0;
    //float max_rho = 0.0;
    //double max_cnt = 0;
    for(int t = 1; t <= numangle; ++t) {
        for(int r = 1; r <= numrho; ++r) {
            int base = t * (numrho+2) + r;
            if(accum[base] > threshold &&
                    accum[base] > accum[base-1] &&
                    accum[base] > accum[base+1] &&
                    accum[base] > accum[base-numrho-2] &&
                    accum[base] > accum[base+numrho+2]) {
                float theta = theta_vec[t-1];
                float rho = (r-1-zero_rho_idx)* rho_resolution;
                lines.push_back({rho, theta});
            }
            //if(accum[base] > max_cnt) {
                //max_cnt = accum[base];
                //max_theta = theta_vec[t-1];
                //max_rho = (r-1-zero_rho_idx)* rho_resolution;
            //}
        }
    }

    /* debug */
    //for(int t = 1; t <= numangle; ++t) {
        //int max_cnt = 0;
        //double rho_best = 0.0;
        //for(int r = 1; r <= numrho; ++r) {
            //int base = t * (numrho+2) + r;
            //if(accum[base] > max_cnt) {
                //max_cnt = accum[base];
                //rho_best = (r-1-zero_rho_idx)* rho_resolution;
            //}
        //}
        //cout << "theta:" << theta_vec[t-1] << ", max_cnt=" << max_cnt << ", rho_best=" << rho_best << '\n';
    //}
    /* ------*/
    //cout << "range_hough: \n"; 
    //SHOW(max_cnt);
    //SHOW(max_theta);
    //SHOW(max_rho);
    //for(int ro = 0; ro < numrho; ++ro) {
        //for(int th = 0; th < numangle; ++th) {
            //int base = (th+1) * (numrho+2) + (ro+1);
            //if(accum[base] > threshold &&
                    //accum[base] > accum[base-1] &&
                    //accum[base] > accum[base+1] &&
                    //accum[base] > accum[base-numrho-2] &&
                    //accum[base] > accum[base+numrho+2]) {
                ////base_vec.push_back(base);
                //float theta = theta_vec[th];
                //float rho = (ro - zero_rho_idx) * rho_resolution;
                //lines.push_back({rho, theta});
            //}
        //}
    //}

    //const double scale = 1./(numrho+2);
    //for(int base : base_vec) {
        //int theta_id = cvFloor(base * scale) - 1;
        //int rho_id = base - (theta_id+1) * (numrho+2) -1;
        //float theta = theta_vec[theta_id];
        //float rho = (rho_id - zero_rho_idx) * rho_resolution;
        //lines.push_back({rho, theta});
    //}
    delete[] tabCos;
    delete[] tabSin;
    delete[] accum;
    return true;
}


