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



//����һ���й�����λ����Ϣ�Ľṹ��
typedef struct center_info {
	cv::Point centerPt;
	double centerScore;
}C_Info;

// �Զ��������Ƚ�����
bool myCompare(C_Info &v1, C_Info &v2);

//����һ����ȡĿ��λ�õ���
class getTargetPosition {
public:
	//�������1��ԭͼ������и��ͼ��
	//�������2���ͼ��ͼ��ͬ���ߴ��С�Ĳ������ϰ��Ĵ��ʹ�ͼ����������mask
	//�������3�������д洢�Ĳ�������
	//				0:��С�������ȣ�С�ڴ��������ȵ�Ŀ�걻���˵���һ��Ϊ�����������ȵ�1/4�����ù�����ܻὫ��ȷĿ����˵������ù�С�����ܻὫ�ϴ�ĸ���Ŀ�걣����
	//				1:��С�����С�����ڹ�������Ӱ�죨һ��Ϊ����Ŀ�������1/4�����ù��󣬿��ܻ���˵�׼ȷĿ�꣬���ù�С������ڽ϶�����Ӱ�죩
	//				2:��ʴ������ֹ����С�����һ��Ϊ����Ŀ�������1/2,���ù��󣬿����޷�ͨ����ʴ���ص�Ŀ����룬���ù�С���ܻ�Ӱ��ʱ�䣩
	//				3:����Ҫ��HSVͨ��id������id����ȡ����Ҫ��ͨ��ͼ��0��H��1��S��2��V��
	//				4:����Ҫ�Ľ��ؾ�����ֵ�����Ϊ�ϰ��Ŀ�ȵĴ��һ�룬����ʱҪ��С��һ����룬����ʱʹ�õ�Ϊ70
	//				5:�ǵ�������Ҫ�Ĵ��ڳߴ��С���ϴ�Ĵ��ڿ��Լ���ǵ㣬С���ڼ��ϸС�Ľǵ㣬����ʹ�õ�2
	//				6:���ǵ���Ӧֵ�ı���ratio��������˵�С��maxPixel*ratio�����أ���ΧΪ��0.01~0.5��������ʱʹ��0.1����������Сһ�㣬������ǵ�
	//				7:�ǵ�Ǽ���ֵ���Ƶ����ƾ��룬�ڴ˾�����ֻ����һ���ǵ㣨����ʱʹ�õ���80�����Ϊ�����ϰ��Ľ��ؾ��볤�ȣ����㷽ʽ���Ϊ�ϰ��ĳ���-��ȣ������ù�����ܽ��ص�Ŀ����˵�һ�������ù�С���ܻ���һ��Ŀ���ϼ�������λ�á�
	//�������4���洢Ŀ��λ�õ���������洢Ŀ���λ��
	//�������5������ԭͼ��Ŀ����汾��������λ����ͼ���ϻ���
	//void targetPosition(IplImage* IplImg, IplImage* toMaskIplImg, double* fParams,
	//	std::vector<cv::Point>& vTargetPos, IplImage* drawIplImg);
	void targetPosition(cv::Mat &img,cv::Mat &toMaskImg, 
	double *fParam,std::vector<cv::Point>&vTargetPos,cv::Mat &drawCenter);
protected:
	//��ȡĿ�������ͼ
	std::vector<cv::Mat> connect_get_target(cv::Mat& img, double singleArea, double areaThre);

	//����mask
	cv::Mat doMask(cv::Mat& img);

	//����ǰ�����������λ��
	cv::Point getCenter(cv::Mat& img);

	//ȥ����ֵͼ��ı�������
	cv::Mat eliminate_bin_background(cv::Mat target, cv::Mat mask, double numsThre);

	//ȥ���Ҷ�ͼ��ı�������
	cv::Mat eliminate_gray_background(cv::Mat target, cv::Mat mask);

	//ȥ��ͼ��ı�������
	cv::Mat eliminate_background(cv::Mat target, cv::Mat mask, double numsThre);

	//��ͨ�������������������˵���ֻ����Ŀ��ͼ��
	cv::Mat setTarget(cv::Mat &filterImg, double  areaThre);

	//����Ŀ��ͼ��Ľ��ؾ���
	cv::Mat calcuDist(cv::Mat &targetImg, double distThre);

	//���ýǵ��ⷽ���Խ��ؾ���ͼ����м��ǵ�
	cv::Mat detectCorner(cv::Mat &distImg,  double blockSize, double ksize, double cornerRatio);

	//�Ǽ���ֵ���Ʒ����Խǵ����ɸѡ
	std::vector<NMS::CornerPoint> NMSCornerPoint(cv::Mat &normalizedCorners,  double nmsThre);

	//���ݽǵ��ȡ���ؾ��������λ��
	std::vector<C_Info> getCornerCenter(cv::Mat &distImg, std::vector<NMS::CornerPoint> filteredCornerPoints);

	//���㱳����ĳ�㵽ǰ����Ե�Ͼ�������ĵ�
	cv::Point calcuMinDist_of_edge(cv::Point2f &pt, cv::Mat &img);

	//�����ĵ�λ�ð��սǵ���Ӧǿ�Ƚ������򣬻�ȡ���յ�ץȡ˳��
	std::vector<C_Info> getSortCenter(std::vector<std::vector<C_Info>>vC_Infos);

	//��ԭͼ���ϻ�������λ�ò����ݵ���Ϣǿ�Ƚ��б��
	void drawTargetInfo(cv::Mat &drawImg, std::vector<C_Info>vC_infos);

	//��������֮��ľ���
	double calculateDistance(const cv::Point& point1, const cv::Point& point2);

	//����ֱ�߼��
	void hough_detect_line(cv::Mat &image,cv::Mat mask,cv::Mat filledImg);
};
