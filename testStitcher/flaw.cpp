//#include"myFlaw.h"
//
////��ȡͼ��
//void readImage(cv::Mat &img)
//{
//	img = cv::imread("D:\\Ŀ��ƥ����ʶ��\\ȱ��\\����A\\0001-001.jpg");
//}
////��������ȱ��
//void findBreakage(cv::Mat &img)
//{
//	//1.�������ͼ��תΪ��ͨ��ͼ��(�Ҷ�ͼ)
//	//cv::Mat grayImg;
//	//cv::cvtColor(img, grayImg, COLOR_BGR2GRAY);
//
//	//2.�ԻҶȻ����ͼ�������ֵ�˲���Ϊ����ȱ�ڼ����׼����
//	cv::Mat blurImg(img.size(), img.type());
//	cv::medianBlur(img, blurImg, 309);
//
//	//3.�Ҷ�ͼ�����˲�֮���ͼ�����Ȼ����ж�ֵ������
//	cv::Mat subtractImg(blurImg.size(), blurImg.type());
//	cv::absdiff(img, blurImg, subtractImg);//���������ֵ 
//	cv::Mat threImg;
//	//cv::threshold(subtractImg, threImg, 70, 255, cv::THRESH_BINARY);
//	//��ͼ����ж�ֵ������
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
////�����̻���
//void connectAll()
//{
//	cv::Mat img;
//	readImage(img);
//	//�и��������
//	cv::Mat rectImg;
//	getWhorl(img, rectImg,360,168);
//	findBreakage(rectImg);
//}