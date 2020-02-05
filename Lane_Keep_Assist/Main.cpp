#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;

int main()
{
    Mat img = imread("C:\\Users\\jaiso\\Documents\\Github\\Lane_Keep_Assist\\Lane_Keep_Assist\\Resources\\test_images\\solidWhiteCurve.jpg");
    namedWindow("image", WINDOW_NORMAL);
    imshow("image", img);
    waitKey(0);
    return 0;
}
