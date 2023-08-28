//#include"Stitcher.h"
//
//int main() {
//	std::vector<cv::Mat>vStitcherImg;		//用来存储拼接后的图像
//	int overallWidth = 0;					//用来存放一个完整周期的宽度
//	int tempWidth = 0;					//选取的模板宽度，为后续周期调整做准备
//	for (int j = 13; j < 36; j++)
//	{
//		//预处理，获取图像
//
//		std::vector<cv::Mat>vImg;		//建立容器存储采集的图像，六张为一组，一个相机采集两张
//		vImg.resize(6);
//		string imagePath1, imagePath2, imagePath3;		//获取路径
//		for (int i = 0; i < 2; i++)
//		{
//			//double start0 = (double)getTickCount();
//			imagePath1 = GetFileName("D:\\files\\testImage\\SaveImage-正常-5.24\\螺纹检测\\合格\\0", j, i);
//			imagePath2 = GetFileName("D:\\files\\testImage\\SaveImage-正常-5.24\\螺纹检测\\合格\\1", j, i);
//			imagePath3 = GetFileName("D:\\files\\testImage\\SaveImage-正常-5.24\\螺纹检测\\合格\\2", j, i);
//
//			//1.读取图像
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
//		//进行螺纹拼接
//		int arrAngle[6] = { 60,59,60,61,60,60 };
//		int arrCutX[6] = { 330,326,332 };
//		int width = 100;
//		DoStitcher st;
//		double start = (double)getTickCount();
//		cv::Mat resultImg=st.Image_Stitching(vImg, arrCutX, width, arrAngle);
//        double end = (double)getTickCount();
//		double time = 1000 * (end - start) / getTickFrequency();
//		cout << "耗时：" << time << " ms" << endl;
//		cv::imshow("拼接图像", resultImg);
//		cv::waitKey(2000);
//	}
//}