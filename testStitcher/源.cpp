//#include"Stitcher.h"
//
//int main() {
//	std::vector<cv::Mat>vStitcherImg;		//�����洢ƴ�Ӻ��ͼ��
//	int overallWidth = 0;					//�������һ���������ڵĿ��
//	int tempWidth = 0;					//ѡȡ��ģ���ȣ�Ϊ�������ڵ�����׼��
//	for (int j = 13; j < 36; j++)
//	{
//		//Ԥ������ȡͼ��
//
//		std::vector<cv::Mat>vImg;		//���������洢�ɼ���ͼ������Ϊһ�飬һ������ɼ�����
//		vImg.resize(6);
//		string imagePath1, imagePath2, imagePath3;		//��ȡ·��
//		for (int i = 0; i < 2; i++)
//		{
//			//double start0 = (double)getTickCount();
//			imagePath1 = GetFileName("D:\\files\\testImage\\SaveImage-����-5.24\\���Ƽ��\\�ϸ�\\0", j, i);
//			imagePath2 = GetFileName("D:\\files\\testImage\\SaveImage-����-5.24\\���Ƽ��\\�ϸ�\\1", j, i);
//			imagePath3 = GetFileName("D:\\files\\testImage\\SaveImage-����-5.24\\���Ƽ��\\�ϸ�\\2", j, i);
//
//			//1.��ȡͼ��
//			cv::Mat img1 = cv::imread(imagePath1,0);
//			cv::Mat img2 = cv::imread(imagePath2,0);
//			cv::Mat img3 = cv::imread(imagePath3,0);
//
//			if (img1.empty() || img2.empty() || img3.empty())
//			{
//				cout << "image open error" << endl;
//
//			}
//			vImg[0 + i] = img1;
//			vImg[2 + i] = img3;
//			vImg[4 + i] = img2;
//		}
//
//		//��������ƴ��
//		int arrAngle[6] = { 60,59,60,61,60,60 };
//		int arrCutX[6] = { 330,326,332 };
//		int width = 100;
//		DoStitcher st;
//		double start = (double)getTickCount();
//		cv::Mat resultImg=st.Image_Stitching(vImg, arrCutX, width, arrAngle);
//        double end = (double)getTickCount();
//		double time = 1000 * (end - start) / getTickFrequency();
//		cout << "��ʱ��" << time << " ms" << endl;
//		cv::imshow("ƴ��ͼ��", resultImg);
//		cv::waitKey(2000);
//	}
//}