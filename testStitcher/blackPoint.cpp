//#include"myBlack.h"
//
//void findBlack(cv::Mat &img)
//{
//	img = cv::imread("D:\\files\\testImage\\���Ʋɼ�ͼ��\\�ڵ�\\����C���\\�ϸ�\\0\\0004-001.jpg");
//	if (img.empty())
//			{
//						cout << "image open error" << endl;
//					
//			}
//	cv::Mat rectImg;
//	getWhorl(img, rectImg,360,168);			//�и����ͼ��
//
//	//2.�ԻҶȻ����ͼ�������ֵ�˲���Ϊ����ȱ�ڼ����׼����
//	cv::Mat blurImg(rectImg.size(), rectImg.type());
//	cv::medianBlur(rectImg, blurImg, 209);
//
//	//3.�Ҷ�ͼ�����˲�֮���ͼ�����Ȼ����ж�ֵ������
//	cv::Mat subtractImg(blurImg.size(), blurImg.type());
//	cv::absdiff(rectImg, blurImg, subtractImg);//���������ֵ 
//
//	//���ж�ֵ������
//	cv::Mat threImg;
//	cv::threshold(subtractImg, threImg, 80, 255, cv::THRESH_BINARY_INV);
//
//	////��������Ӧ��ֵ��
//	//cv::Mat threImg;
//	//int blockSize = 15;
//	//int constValue = 10;
//	//cv::Mat threImage;
//	//cv::adaptiveThreshold(subtractImg, threImage, 255, cv::ADAPTIVE_THRESH_MEAN_C,
//	//	cv::THRESH_BINARY, blockSize, constValue);
//}