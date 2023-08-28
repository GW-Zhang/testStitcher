//#include"myBlack.h"
//
//void findBlack(cv::Mat &img)
//{
//	img = cv::imread("D:\\files\\testImage\\螺纹采集图像\\黑点\\螺纹C检测\\合格\\0\\0004-001.jpg");
//	if (img.empty())
//			{
//						cout << "image open error" << endl;
//					
//			}
//	cv::Mat rectImg;
//	getWhorl(img, rectImg,360,168);			//切割出子图像
//
//	//2.对灰度化后的图像进行中值滤波（为后续缺口检测做准备）
//	cv::Mat blurImg(rectImg.size(), rectImg.type());
//	cv::medianBlur(rectImg, blurImg, 209);
//
//	//3.灰度图像与滤波之后的图像做差，然后进行二值化操作
//	cv::Mat subtractImg(blurImg.size(), blurImg.type());
//	cv::absdiff(rectImg, blurImg, subtractImg);//做差求绝对值 
//
//	//进行二值化操作
//	cv::Mat threImg;
//	cv::threshold(subtractImg, threImg, 80, 255, cv::THRESH_BINARY_INV);
//
//	////进行自适应二值化
//	//cv::Mat threImg;
//	//int blockSize = 15;
//	//int constValue = 10;
//	//cv::Mat threImage;
//	//cv::adaptiveThreshold(subtractImg, threImage, 255, cv::ADAPTIVE_THRESH_MEAN_C,
//	//	cv::THRESH_BINARY, blockSize, constValue);
//}