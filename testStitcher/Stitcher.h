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
//#define PI 3.1415926   //����Բ����PIΪ3.1415926
//
////���ļ��ж�ȡͼ��
//char * GetFileName(const char * dir, int i, int j);
//
////����һ��ƴ����
//class DoStitcher {
//public:
//	/*ƴ�Ӻ���
//	
//	*/
//	cv::Mat Image_Stitching(std::vector<cv::Mat>vImg, int *cutX, int width, int *arrAngle);
//
//protected:
//	//1.ͼ�������������ɼ���ͼ�����Ʋ��ֿ��ܻ������б�����
//	void image_rectification(std::vector<cv::Mat>vImg, std::vector<cv::Mat>&adjImg, int width);
//
//	//1.1 Ѱ�ҽ�����Ҫ��������
//	void findFeaturePoints(cv::Mat &img, int cutX, int width, cv::Point &pt1, cv::Point &pt2);
//
//	//1.2 ��ͼ����нǶ�����
//	void adjustImg(cv::Mat &img, cv::Mat &adjImg, cv::Point pt1, cv::Point pt2);
//
//	//2.ͼ���и���������������и��
//	//ͨ���Զ����ѡ����������֮����������ȡ����
//	void getWhorl(std::vector<cv::Mat>adjImg, std::vector<cv::Mat>&rectImg, int *arrCutX, int width);
//
//	//2.1 //�������ͼƬ��ʱ����ת
//	void rotate(cv::Mat &image, cv::Mat &rectImage, int angle);
//
//	//3.�����Ʋ��ֽ�������չ��
//	void speardCylinder(std::vector<cv::Mat>rectImg,std::vector<cv::Mat>&speardImg,int radius);
//	//3.1 �������չ������
//	void speardCylinderMatrix(cv::Mat &rectImage01, int &radius, cv::Mat &mapX, cv::Mat &mapY);
//	//3.2 ������ͶӰ������չ��
//
//	//4.ƴ�Ӻ���
//	void getMostCorr(std::vector<cv::Mat>&vImg, cv::Mat &allImg, int *arr, int &overallWidth, int &tempWidth);
//	//4.1 //��ȡģ��ƥ����ض�
//	void getCorr(cv::Mat &img1, cv::Mat &img2, cv::Mat &matchResult, int angle);
//	//4.2 //�õ����ƥ�������ƴ��
//	void StitImg(cv::Mat &img1, cv::Mat &img2, cv::Mat &dstImg, int angle, int BestX, int BestY);
//
//	//5.��һ����β�и����ȷ�����ڣ���ƴ��ͼ�������β�иʹ����һ������
//	void getPeriod(cv::Mat &img01, cv::Mat &img02, int angle, int &dis, cv::Point &p);
//
//
//
//private:
//	//��������
//	int cutAdjX = 400;		//�ڵ�һ���ֽ���ʱ�����ڸ���һ����ʼx�и�λ�ã����ٽ�������
//	int offsetX = 5;		//������Ѱ����������Ϊ��������ʱ����Ѱ�ҵ�������������ƫ��x�����أ���ֹ�ܸ���
//	int topX = 300;
//	int bottomX = 600;		//������������ֻ����topX~bottomX�ڵ����򣬴����������������
//	int adjWidth = 80;
//
//	//�и��
//	int SpeardRadius = 178;		//����չ���İ뾶���˰뾶�����ֶ�����
//	int addWidth = 55;			//����������������и�ֿ��ں���ƥ��ʱ���������ƶ����֣���֤����ƥ��׼ȷ��
//
//	//ƴ�Ӳ���
//	int offsetY = 3;			//��ƥ��ʱ����������ƥ��λ�õ�����offsetY��������ƫ��Ѱ�����ƥ��λ��
//	int doCutY = 40;			//�����и�ģ��ĸ߶ȣ�ǰ�����õ��и���������addWidth����ˣ��˴�doCutYӦ��С��addWidth
//	int bottomCutY = 45;		//�и�����ޣ���֤���������ʱ���������ƶ�
//	int tempWidth = 20;			//����ƥ��ģ��Ŀ��
//
//};
//
