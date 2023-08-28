//#pragma once
//#include<iostream>
//#include<cmath>
//#include<opencv2/opencv.hpp>
//#include<stdio.h>
//#include<string>
//#include<opencv2/features2d/features2d.hpp>
//#include<ctime>
//#include<thread>
//#include<map>
//#include<list>
//#include<mutex>
//#include<future>
//#include<atomic>
//
//using namespace std;
//using namespace cv;
//
//#define PI 3.1415926   //定义圆周率PI为3.1415926
//
////在文件中读取图像
//char * GetFileName(const char * dir, int i, int j);
//
////定义一个拼接类
//class DoStitcher {
//public:
//	/*拼接函数
//	
//	*/
//	cv::Mat Image_Stitching(std::vector<cv::Mat>vImg, int *cutX, int width, int *arrAngle);
//
//protected:
//	//1.图像修正函数（采集的图像螺纹部分可能会出现歪斜情况）
//	void image_rectification(std::vector<cv::Mat>vImg, std::vector<cv::Mat>&adjImg, int width);
//
//	//1.1 寻找矫正需要的特征点
//	void findFeaturePoints(cv::Mat &img, int cutX, int width, cv::Point &pt1, cv::Point &pt2);
//
//	//1.2 对图像进行角度修正
//	void adjustImg(cv::Mat &img, cv::Mat &adjImg, cv::Point pt1, cv::Point pt2);
//
//	//2.图像切割函数（将螺纹区域切割处理）
//	//通过自定义框，选中螺纹区域，之后将子区域提取出来
//	void getWhorl(std::vector<cv::Mat>adjImg, std::vector<cv::Mat>&rectImg, int *arrCutX, int width);
//
//	//2.1 //将输入的图片逆时针旋转
//	void rotate(cv::Mat &image, cv::Mat &rectImage, int angle);
//
//	//3.对螺纹部分进行柱体展开
//	void speardCylinder(std::vector<cv::Mat>rectImg,std::vector<cv::Mat>&speardImg,int radius);
//	//3.1 求得螺纹展开矩阵
//	void speardCylinderMatrix(cv::Mat &rectImage01, int &radius, cv::Mat &mapX, cv::Mat &mapY);
//	//3.2 利用重投影，进行展开
//
//	//4.拼接函数
//	void getMostCorr(std::vector<cv::Mat>&vImg, cv::Mat &allImg, int *arr, int &overallWidth, int &tempWidth);
//	//4.1 //获取模板匹配相关度
//	void getCorr(cv::Mat &img1, cv::Mat &img2, cv::Mat &matchResult, int angle);
//	//4.2 //得到最佳匹配点后进行拼接
//	void StitImg(cv::Mat &img1, cv::Mat &img2, cv::Mat &dstImg, int angle, int BestX, int BestY);
//
//	//5.做一个首尾切割函数，确定周期，对拼接图像进行首尾切割，使其在一个周期
//	void getPeriod(cv::Mat &img01, cv::Mat &img02, int angle, int &dis, cv::Point &p);
//
//
//
//private:
//	//矫正部分
//	int cutAdjX = 400;		//在第一部分矫正时，用于给定一个初始x切割位置，减少矫正区域
//	int offsetX = 5;		//用于在寻找特征点作为矫正依据时，将寻找到的特征点向内偏移x个像素，防止受干扰
//	int topX = 300;
//	int bottomX = 600;		//给定矫正区域，只矫正topX~bottomX内的区域，此区域包含螺纹区域
//	int adjWidth = 80;
//
//	//切割部分
//	int SpeardRadius = 178;		//设置展开的半径，此半径自主手动设置
//	int addWidth = 55;			//设置在螺纹区域多切割部分宽，在后面匹配时，允许其移动部分，保证后续匹配准确性
//
//	//拼接部分
//	int offsetY = 3;			//在匹配时，允许其在匹配位置的左右offsetY个像素内偏移寻找最佳匹配位置
//	int doCutY = 40;			//定义切割模板的高度，前面设置的切割区域多出来addWidth，因此，此处doCutY应该小于addWidth
//	int bottomCutY = 45;		//切割的下限，保证其可以搜索时可以上下移动
//	int tempWidth = 20;			//定义匹配模板的宽度
//
//};
//
