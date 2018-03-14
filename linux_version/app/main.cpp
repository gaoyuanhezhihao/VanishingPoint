#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include "vanish.hpp"
#include "Config.hpp"
#include "base.hpp"

#define PI 3.1415926
#define CENTER_RANGE 0.1

using namespace std;
using namespace cv;

int main(int argc, const char ** argv)
{
	Point mark_point, current_point;
	int center_x_min, center_x_max, center_y_min, center_y_max;
	Mat image;

	Mat dst, cdst, edge;
	//double start = double(getTickCount());

    if(argc != 2) {
        cout << "Error! \nusage example ./bin/vanish_pt ../data/video.wmv\n";
    }
    configs.init(argv[1]);
    const string dst_dir = configs["result_dir"];
    ImgLogger edge_log(dst_dir, string("edge"));
    ImgLogger line_rgb(dst_dir, string("line_rgb"));
    ImgLogger line_edge(dst_dir, string("line_edge"));
    ImgLogger rgb_log(dst_dir, string("rgb"));
	VideoCapture cap(string(configs["video_path"]).c_str());
	if (!cap.isOpened())
	{
		cout << "Capture could not be opened successfully" << endl;
		return -1;
	}
	//namedWindow("Video");
	char order = 0;

	//get the initial position image.
    int id = 0;
    double time_elaps = 0.;
	while (1)
	{
        Mat image, dst, cdst, edge;
		cap >> image;
        ++id;
		mark_point = vanish_point_detection(image, cdst, edge, time_elaps);
        cv::imshow("init", image);
        waitKey(10);
		cout << "what about this ?" << endl;
		std::cin >> order;
		if (order == 'y')
		{
			break;
		}
	}
	center_x_max = (1 + CENTER_RANGE) * mark_point.x;
	center_x_min = (1 - CENTER_RANGE) * mark_point.x;
	center_y_max = (1 + CENTER_RANGE) * mark_point.y;
	center_y_min = (1 - CENTER_RANGE) * mark_point.y;
	
    double cnt = 0;
    double average = 0.0;
	while (char(waitKey(1)) != 'q' && cap.isOpened())
	{
        Mat image, dst, cdst, edge;
		cap >> image;
		if (image.empty())
		{
			cout << "Video over" << endl;
			break;
		}
		else
		{
            rgb_log.save(image, id);
			current_point = vanish_point_detection(image, cdst, edge, time_elaps);
			//Canny(image, dst, 30, 70, 3);
			//cvtColor(dst, cdst, CV_GRAY2BGR);

			//vector<Vec2f> lines;
			//// detect lines
			//HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);

			//// draw lines
			//for (size_t i = 0; i < lines.size(); i++)
			//{
			//	float rho = lines[i][0], theta = lines[i][1];
			//	Point pt1, pt2;
			//	double a = cos(theta), b = sin(theta);
			//	double x0 = a*rho, y0 = b*rho;
			//	pt1.x = cvRound(x0 + 1000 * (-b));
			//	pt1.y = cvRound(y0 + 1000 * (a));
			//	pt2.x = cvRound(x0 - 1000 * (-b));
			//	pt2.y = cvRound(y0 - 1000 * (a));
			//	if (10 * PI / 180 < theta && theta < 60 * PI / 180)
			//	{
			//		line(cdst, pt1, pt2, Scalar(0, 255, 0), 3, CV_AA);
			//	}
			//	else if (110 * PI / 180 < theta && theta < 170 * PI / 180)
			//	{
			//		line(cdst, pt1, pt2, Scalar(255, 0, 0), 3, CV_AA);
			//	}
			//	else
			//	{
			//		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
			//	}
			//}
            average = average * (cnt/(cnt+1)) + time_elaps /(cnt+1);
            ++cnt;
			cout << "It took " << time_elaps << " ms." << endl;
			if (current_point.x > center_x_max)
			{
				cout << "left" << endl;
			}
			else if (current_point.x < center_x_min)
			{
				cout << "right" << endl;
			}
			else
			{
				cout << "center" << endl;
			}
			line(cdst, Point(mark_point.x, 0), Point(mark_point.x, image.rows), Scalar(100, 100, 100), 1, CV_AA);
			//imshow("detected lines", cdst);
            edge_log.save(edge, id);
            line_edge.save(cdst, id);
            line_rgb.save(image, id);
            ++id;

            //imwrite(dst_dir+to_string(int(cnt))+".jpg", cdst);
            //++idx;
            //std::string dst_path = dst_dir+std::to_string(idx)+string(".jpg");
            //imwrite(dst_path, cdst);
		}

	}
	cout << "Press any key to exit:" << endl;
	std::cin >> order;



	cv::waitKey();
    cout << "average time used for one frame:" << average << " ms\n";
	return 0;
}
