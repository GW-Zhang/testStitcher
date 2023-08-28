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
#define PI acos(-1)   //����Բ����PIΪ3.14152
//�ѵ�����Ϊ����ʵ�־�γ�Ⱥ�ī����ͶӰ�ĺ���
using namespace std;
using namespace cv;
//���ļ��ж�ȡͼ��
char * GetFileName(const char * dir, int i, int j);

//��˹�������²���
cv::Mat paramidGaussImage(cv::Mat &src, int paramidNums);

//��ȡƴ��λ��
void getPyrMatch(cv::Mat &speImage01, cv::Mat &speImage02, int angle, cv::Point pt, cv::Point maxLoc);

//�Ը�˹��������������ƥ��ͼ�����ƴ��
void StitcherImg(cv::Mat &pyrSpeImg01, cv::Mat &pyrSpeImg02, cv::Mat &speImage01, cv::Mat &speImage02, cv::Mat& dstImg, int angle);

//����ת��
void speardCylinderMatrix(cv::Mat &rectImage01, int &radius, cv::Mat &mapX, cv::Mat &mapY);

//��ת
void rotate(cv::Mat &image, cv::Mat &rectImage, int angle);



//����ģ��ƥ�䣬����ͼ��ƴ��
void getStitcherImage(cv::Mat &speImage01, cv::Mat &speImage02, cv::Mat &dst, int angle);

//�����·�������������Ѱ��
void findFeaturePoints(cv::Mat &img, int cutX, int width, cv::Point &pt1, cv::Point &pt2);






//ͨ���Զ����ѡ����������֮����������ȡ����
void getWhorl(cv::Mat &image, cv::Mat &rectImage, int x, int width);

//�����к�����������
void resultImg(std::vector<cv::Mat>&vImg, int num, int *arr);



//��һ����β�и����ȷ�����ڣ���ƴ��ͼ�������β�иʹ����һ������
void getPeriod(cv::Mat &img01, cv::Mat &img02, int angle, int &dis, cv::Point &p);

//����ͼ�����ƴ���ں�
void mulStitcherImage(std::vector<cv::Mat>&vSpeImg, cv::Mat &dst, int num, int *arr);



//������ƴ��ͼ����е�������ͬ�����£���ʼ��������ͬ��
void adjPeriod(cv::Mat &target, cv::Mat &img, int overallWidth, cv::Mat &adjImg);

//ȡ�����е��и���ͼ�񣬽�������һ��������
void allWhorl(int *arr01, std::vector<cv::Mat>&vAllimg);

//��ȡͼ��
void allStitcher(std::vector<cv::Mat>&vImg, cv::Mat &dst, int *arr, int cutAdjX, int *arrCutX, int width);

//���ȼ���ͼ���������
void detectPoints(cv::Mat &img, cv::Point &pt1, cv::Point &pt2, int X);

//��ͼ����нǶ�����
void adjustImg(cv::Mat &img, cv::Mat &adjImg, cv::Point &pt1, cv::Point &pt2);

//ƴ��ȫ��ͼ��
void getImg(std::vector<cv::Mat>&vImg, int *arr, cv::Mat &dst);

//����ʹ�ö�ģ�����ƥ��
void mulTempMatch(cv::Mat &speImage01, cv::Mat &speImage02, cv::Mat& dstImg, int angle);

//Ѱ��������
void getMostCorr(std::vector<cv::Mat>&vImg, cv::Mat &allImg, int *arr, int &overallWidth, int &tempWidth);

//��ȡģ��ƥ����ض�
void getCorr(cv::Mat &img1, cv::Mat &img2, cv::Mat &matchResult, int angle);

//�õ����ƥ�������ƴ��
void StitImg(cv::Mat &img1, cv::Mat &img2, cv::Mat &dstImg, int angle, int BestX, int BestY);

#ifndef tic
#define tic LARGE_INTEGER litmp; int qpart1, qpart2; double dfm, dff, dft; QueryPerformanceFrequency(&litmp); dff = (double)litmp.QuadPart; QueryPerformanceCounter(&litmp); qpart1 = (int)litmp.QuadPart; 
#endif

#ifndef toc
#define toc  QueryPerformanceCounter(&litmp); qpart2 = (int)litmp.QuadPart; dfm = (double)(qpart2 - qpart1); dft = dfm / dff; printf("%fms\n", dft * 1000);
#endif
