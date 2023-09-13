#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

namespace NMS {
	struct CornerPoint {
		cv::Point2f location;
		float response;
	};

	bool CompareResponse(const CornerPoint& a, const CornerPoint& b);

	std::vector<CornerPoint> PerformNMS(const std::vector<CornerPoint>& cornerPoints, float threshold);
}



//定义一个有关中心位置信息的结构体
typedef struct center_info {
	cv::Point centerPt;
	double centerScore;
}C_Info;

// 自定义容器比较类型
bool myCompare(C_Info &v1, C_Info &v2);

//定义一个获取目标位置的类
class getTargetPosition {
public:
	//输入参数1：原图像或者切割的图像
	//输入参数2：和检测图像同样尺寸大小的不包含料包的传送带图像，用来制作mask
	//输入参数3：数组中存储的参数含义
	//				0:最小轮廓长度，小于此轮廓长度的目标被过滤掉（一般为单个轮廓长度的1/4，设置过大可能会将正确目标过滤掉，设置过小，可能会将较大的干扰目标保留）
	//				1:最小面积大小，用于过滤噪声影响（一般为单个目标面积的1/4，设置过大，可能会过滤掉准确目标，设置过小，会存在较多杂物影响）
	//				2:腐蚀操作终止的最小面积（一般为单个目标面积的1/2,设置过大，可能无法通过腐蚀将重叠目标分离，设置过小可能会影响时间）
	//				3:所需要的HSV通道id，输入id，提取所需要的通道图像（0：H，1：S，2：V）
	//				4:所需要的近地距离阈值，大概为料包的宽度的大概一半，设置时要略小于一半距离，测试时使用的为70
	//				5:角点检测所需要的窗口尺寸大小，较大的窗口可以检测大角点，小窗口检测细小的角点，测试使用的2
	//				6:最大角点响应值的比率ratio，例如过滤掉小于maxPixel*ratio的像素，范围为（0.01~0.5），测试时使用0.1，尽量设置小一点，检测多个角点
	//				7:角点非极大值抑制的抑制距离，在此距离内只保留一个角点（测试时使用的是80，大概为单个料包的近地距离长度，计算方式大概为料包的长度-宽度），设置过大可能将重叠目标过滤掉一个，设置过小可能会在一个目标上检测出两个位置。
	//输入参数4：存储目标位置点的容器，存储目标的位置
	//输入参数5：输入原图像的拷贝版本，将中心位置在图像上画出
	//void targetPosition(IplImage* IplImg, IplImage* toMaskIplImg, double* fParams,
	//	std::vector<cv::Point>& vTargetPos, IplImage* drawIplImg);
	void targetPosition(cv::Mat &img,cv::Mat &toMaskImg, 
	double *fParam,std::vector<cv::Point>&vTargetPos,cv::Mat &drawCenter);
protected:
	//获取目标的区域图
	std::vector<cv::Mat> connect_get_target(cv::Mat& img, double singleArea, double areaThre);

	//制作mask
	cv::Mat doMask(cv::Mat& img);

	//计算前景区域的中心位置
	cv::Point getCenter(cv::Mat& img);

	//去除二值图像的背景干扰
	cv::Mat eliminate_bin_background(cv::Mat target, cv::Mat mask, double numsThre);

	//去除灰度图像的背景干扰
	cv::Mat eliminate_gray_background(cv::Mat target, cv::Mat mask);

	//去除图像的背景干扰
	cv::Mat eliminate_background(cv::Mat target, cv::Mat mask, double numsThre);

	//连通域分析，将干扰情况过滤掉，只保留目标图像
	cv::Mat setTarget(cv::Mat &filterImg, double  areaThre);

	//计算目标图像的近地距离
	cv::Mat calcuDist(cv::Mat &targetImg, double distThre);

	//利用角点检测方法对近地距离图像进行检测角点
	cv::Mat detectCorner(cv::Mat &distImg,  double blockSize, double ksize, double cornerRatio);

	//非极大值抑制方法对角点进行筛选
	std::vector<NMS::CornerPoint> NMSCornerPoint(cv::Mat &normalizedCorners,  double nmsThre);

	//根据角点获取近地距离的中心位置
	std::vector<C_Info> getCornerCenter(cv::Mat &distImg, std::vector<NMS::CornerPoint> filteredCornerPoints);

	//计算背景上某点到前景边缘上距离最近的点
	cv::Point calcuMinDist_of_edge(cv::Point2f &pt, cv::Mat &img);

	//对中心点位置按照角点响应强度进行排序，获取最终的抓取顺序
	std::vector<C_Info> getSortCenter(std::vector<std::vector<C_Info>>vC_Infos);

	//在原图像上画出中心位置并根据点信息强度进行标号
	void drawTargetInfo(cv::Mat &drawImg, std::vector<C_Info>vC_infos);

	//计算两点之间的距离
	double calculateDistance(const cv::Point& point1, const cv::Point& point2);

	//霍夫直线检测
	void hough_detect_line(cv::Mat &image,cv::Mat mask,cv::Mat filledImg);
};
