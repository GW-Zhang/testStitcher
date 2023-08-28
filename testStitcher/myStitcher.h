#pragma once
#pragma once

#include<iostream>
#include<cmath>
#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<string>
#include<opencv2/features2d/features2d.hpp>
#include<ctime>
#include<thread>
#include<map>
#include<list>
#include<mutex>
#include<future>
#include<atomic>
#define PI acos(-1)   //定义圆周率PI为3.14152
//把地球视为球体实现经纬度和墨卡托投影的函数
using namespace std;
using namespace cv;
//在文件中读取图像
char * GetFileName(const char * dir, int i, int j);

//高斯金字塔下采样
cv::Mat paramidGaussImage(cv::Mat &src, int paramidNums);

//获取拼接位置
void getPyrMatch(cv::Mat &speImage01, cv::Mat &speImage02, int angle, cv::Point pt, cv::Point maxLoc);

//对高斯金字塔处理过后的匹配图像进行拼接
void StitcherImg(cv::Mat &pyrSpeImg01, cv::Mat &pyrSpeImg02, cv::Mat &speImage01, cv::Mat &speImage02, cv::Mat& dstImg, int angle);

//螺纹转开
void speardCylinderMatrix(cv::Mat &rectImage01, int &radius, cv::Mat &mapX, cv::Mat &mapY);

//旋转
void rotate(cv::Mat &image, cv::Mat &rectImage, int angle);



//利用模板匹配，进行图像拼接
void getStitcherImage(cv::Mat &speImage01, cv::Mat &speImage02, cv::Mat &dst, int angle);

//利用新方法进行特征点寻找
void findFeaturePoints(cv::Mat &img, int cutX, int width, cv::Point &pt1, cv::Point &pt2);






//通过自定义框，选中螺纹区域，之后将子区域提取出来
void getWhorl(cv::Mat &image, cv::Mat &rectImage, int x, int width);

//对所有函数进行整合
void resultImg(std::vector<cv::Mat>&vImg, int num, int *arr);



//做一个首尾切割函数，确定周期，对拼接图像进行首尾切割，使其在一个周期
void getPeriod(cv::Mat &img01, cv::Mat &img02, int angle, int &dis, cv::Point &p);

//多张图像进行拼接融合
void mulStitcherImage(std::vector<cv::Mat>&vSpeImg, cv::Mat &dst, int num, int *arr);



//将所有拼接图像进行调整到相同周期下（开始、结束相同）
void adjPeriod(cv::Mat &target, cv::Mat &img, int overallWidth, cv::Mat &adjImg);

//取出所有的切割子图像，将其存放在一个容器中
void allWhorl(int *arr01, std::vector<cv::Mat>&vAllimg);

//读取图像
void allStitcher(std::vector<cv::Mat>&vImg, cv::Mat &dst, int *arr, int cutAdjX, int *arrCutX, int width);

//首先检测出图像的特征点
void detectPoints(cv::Mat &img, cv::Point &pt1, cv::Point &pt2, int X);

//对图像进行角度修正
void adjustImg(cv::Mat &img, cv::Mat &adjImg, cv::Point &pt1, cv::Point &pt2);

//拼接全部图像
void getImg(std::vector<cv::Mat>&vImg, int *arr, cv::Mat &dst);

//尝试使用多模板进行匹配
void mulTempMatch(cv::Mat &speImage01, cv::Mat &speImage02, cv::Mat& dstImg, int angle);

//寻找最大互相关
void getMostCorr(std::vector<cv::Mat>&vImg, cv::Mat &allImg, int *arr, int &overallWidth, int &tempWidth);

//获取模板匹配相关度
void getCorr(cv::Mat &img1, cv::Mat &img2, cv::Mat &matchResult, int angle);

//得到最佳匹配点后进行拼接
void StitImg(cv::Mat &img1, cv::Mat &img2, cv::Mat &dstImg, int angle, int BestX, int BestY);

#ifndef tic
#define tic LARGE_INTEGER litmp; int qpart1, qpart2; double dfm, dff, dft; QueryPerformanceFrequency(&litmp); dff = (double)litmp.QuadPart; QueryPerformanceCounter(&litmp); qpart1 = (int)litmp.QuadPart; 
#endif

#ifndef toc
#define toc  QueryPerformanceCounter(&litmp); qpart2 = (int)litmp.QuadPart; dfm = (double)(qpart2 - qpart1); dft = dfm / dff; printf("%fms\n", dft * 1000);
#endif
