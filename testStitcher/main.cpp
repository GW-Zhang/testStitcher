#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/features2d/features2d.hpp>
#include"myStitcher.h"
//#include"myFlaw.h"
//#include"myBlack.h"
using namespace std;
//int main()
//{
//
//
//	//std::vector<cv::Mat>vAllimg;
//
//	////存储三个相机每两幅图像之间的夹角参数
//	//int arr01[3] = { 50,75,50 };
//	////	vAllimg.resize(120);
//	//allWhorl(arr01, vAllimg);
//	//	cv::imshow("1", vAllimg[0]);
//
//	//int num = 3;
//
//	////存储不同相机之间的夹角
//	//int arr[2] = { 80,75 };
//	//resultImg(vAllimg, num, arr);
//	cv::Mat img;
//	cv::Mat dst;
//	string imgPath;
//
//
//	//for (int i = 0; i < 60; i++)
//	//{
//	//	imgPath = GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹C检测\\合格\\0", i);
//	//	img = cv::imread(imgPath);
//	//	adjustImg(img, dst,570);
//	//}
//	double start = (double)getTickCount();
//	//adjustImg(img, dst);
//	std::vector<cv::Mat>vImg;
//
//	int arr[6] = { 60,59,60,61,60,60 };
//	int arrCutX[3] = { 330,332,326 };
//	int cutAdjX = 400;
//	int width = 100;
//	allStitcher(vImg, dst, arr, cutAdjX, arrCutX, width);
//	//allStitcher(vImg, arr, dst);
//
//	double end = (double)getTickCount();
//	double time = (end - start) / getTickFrequency();
//
//	cout << "运行时间为：" << time << endl;
//
//	cv::waitKey(0);
//}