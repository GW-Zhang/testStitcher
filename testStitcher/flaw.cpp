//#include"myFlaw.h"
//
////读取图像
//void readImage(cv::Mat &img)
//{
//	img = cv::imread("D:\\目标匹配与识别\\缺口\\螺纹A\\0001-001.jpg");
//}
////分析破损缺口
//void findBreakage(cv::Mat &img)
//{
//	//1.将输入的图像转为单通道图像(灰度图)
//	//cv::Mat grayImg;
//	//cv::cvtColor(img, grayImg, COLOR_BGR2GRAY);
//
//	//2.对灰度化后的图像进行中值滤波（为后续缺口检测做准备）
//	cv::Mat blurImg(img.size(), img.type());
//	cv::medianBlur(img, blurImg, 309);
//
//	//3.灰度图像与滤波之后的图像做差，然后进行二值化操作
//	cv::Mat subtractImg(blurImg.size(), blurImg.type());
//	cv::absdiff(img, blurImg, subtractImg);//做差求绝对值 
//	cv::Mat threImg;
//	//cv::threshold(subtractImg, threImg, 70, 255, cv::THRESH_BINARY);
//	//对图像进行二值化处理
//	int blockSize = 25;
//	int constValue = 10;
//	cv::Mat threImage;
//	cv::adaptiveThreshold(subtractImg, threImage, 255, cv::ADAPTIVE_THRESH_MEAN_C,
//		cv::THRESH_BINARY, blockSize, constValue);
//
//	//4.
//
//
//
//}
//
////将过程汇总
//void connectAll()
//{
//	cv::Mat img;
//	readImage(img);
//	//切割出子区域
//	cv::Mat rectImg;
//	getWhorl(img, rectImg,360,168);
//	findBreakage(rectImg);
//}