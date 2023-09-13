#include "shape_match.h"
#include<math.h>

int main(int argc, char **argv) {

	//用于读取图像所在文件夹以及内部文件
	string str = "1";

	shapematch::ShapeMatching ShapeM("D:\\files\\c++文件夹\\c++ files\\shapeMatching\\shapeMatching", "temp");
	// 读取模板图像
	Mat model = imread("D:\\files\\c++文件夹\\c++ files\\shapeMatching\\shapeMatching\\test\\template\\0-2.bmp");
	//cv::blur(model, model, Size(3, 3));
	// 转灰度
	cv::Mat grayModel;
	cvtColor(model, grayModel, COLOR_BGR2GRAY);

	// 指定要制作的模板角度，尺度范围
	shapematch::AngleRange ar(0.f, 360.f, 1.0f);
	shapematch::ScaleRange sr(1.0f, 1.0f, 0.05f);
	// 开始制作模板,若需要新模板，则重新训练模板
	ShapeM.MakingTemplates(model, ar, sr, 0, 30.f, 70.f);

	// 加载模板
	cout << "Loading model ......" << endl;
	ShapeM.LoadModel();
	cout << "Load succeed." << endl;

	//在文件夹中读入全部图像
	string pattern_bmp =
		"D:\\files\\c++文件夹\\c++ files\\shapeMatching\\shapeMatching\\test\\T";
	std::vector<cv::String> image_files;
	//getFilePaths(image_files, pattern_bmp); //图片排序
	cv::glob(pattern_bmp, image_files, false);
	if (image_files.size() == 0)
	{
		cout << "没有图像" << endl;
		return 0;
	}

	else {

		int MatchNum = 0;
		double timeSum = 0.0;

		for (int i = 1; i < image_files.size(); i++)
		{
			cout << "第" << i << "张测试图像：" << endl;
			cv::Mat source = cv::imread(image_files.at(i));
			//cv::Mat source = cv::imread("test/test01/source_06.bmp");

			if (source.data == 0)
			{
				cout << "未加载进来测试图像" << endl;
				return 1;
			}


			////获取目标图像的roi区域
			//cv::Rect rect = cv::Rect(800, 300, 1000, 1000);
			//source = source(rect);
			Mat draw_source;
			source.copyTo(draw_source);

			//采用通道图像或者灰度图像或者HSV图像
			//source = sourceHSVChannel(source, 1);
			//source=sourceChannel(source, channelNum);		//通道图像
			//cvtColor(source, source, COLOR_BGR2GRAY);		//灰度图像

			//各种滤波处理
			//GaussianBlur(source, source, Size(11, 11), 0, 0);
			//medianBlur(source, source, 15);
			//blur(source, source, Size(9, 9));

			//Timer timer;
			// 开始匹配
			double start = (double)getTickCount();
			auto matches =
				ShapeM.Matching(source, 0.5f, 0.3f, 30.f, 0.9f,
					shapematch::PyramidLevel_2, 4, 0);
			//double t = timer.out("=== Match time ===");
			double time = (((double)getTickCount() - start)) / getTickFrequency() * 1000;
			timeSum += time;
			printf("edge template matching time : %.2f ms\n", time);
			cout << "Final match size: " << matches.size() << endl << endl;
			if (matches.size() != 0)
				MatchNum++;

			// 画出匹配结果
			ShapeM.DrawMatches(draw_source, matches, Scalar(255, 0, 0));

			//cv::imwrite("D:\\files\\c++文件夹\\c++ files\\shapeMatching\\shapeMatching\\result\\"+str+"_"+to_string(i)+".jpg",draw_source);

		}
		double aveTime = timeSum / image_files.size();
		double aveMatch = double(MatchNum) / double(image_files.size());

		cout << "平均耗时为：" << aveTime << " ms" << endl;
		cout << "检测准确度为：" << aveMatch * 100 << "%" << endl;


		system("pause");
		return 0;

	}
	//// 读取搜索图像
	//Mat source = imread("test\\40~50\\test40-50 (38).bmp"); 
	////cv::blur(source, source, Size(3, 3));
	//Mat draw_source;
	//source.copyTo(draw_source);
	//cvtColor(source, source, COLOR_BGR2GRAY);

	//if (source.data == 0)
	//{
	//	cout << "未加载进来测试图像" << endl;
	//	return 1;
	//}

	////自适应二值化
 ////// 自适应阈值二值化
	////Mat srcBinary;
	////cv::adaptiveThreshold(source, source, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 10);
	////		//二值化操作
	////cv::threshold(source, source, 150, 255, THRESH_BINARY);

	//////进行积分图计算
	////cv::Mat sum;
	////cv::integral(source, sum,  CV_32S);
	////cv::normalize(sum, sum, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	//
	////Timer timer;
	//// 开始匹配
	//double start = (double)getTickCount();
	//auto matches =
	//	Kcg.Matching(source, 0.10f, 0.6f, 60.f, 0.9f,
	//		shapematch::PyramidLevel_2,4,10);
	////double t = timer.out("=== Match time ===");
	//double time = (((double)getTickCount() - start)) / getTickFrequency() * 1000;
	//printf("edge template matching time : %.2f ms\n", time);
	//cout << "Final match size: " << matches.size() << endl << endl;

	//// 画出匹配结果
	//Kcg.DrawMatches(draw_source, matches, Scalar( 0, 0, 255));

	//// 画出匹配时间
	//rectangle(draw_source, Rect(Point(0, 0), Point(136, 20)), Scalar(255, 255, 255), -1);
	//cv::putText(draw_source,
	//	"time: " + to_string(time) + "ms",
	//	Point(0, 16), FONT_HERSHEY_PLAIN, 1.f, Scalar(0, 0, 0), 1);

	//// 显示结果图像
	//namedWindow("draw_source", 0);
	//imshow("draw_source", draw_source);
	//waitKey(0);
	//system("pause");
}

//int getMinExp(int length);
//
////画出网格
//int drawGrid(cv::Mat &image)
//{
//	int r = image.rows;
//	int c = image.cols;
//	int exp1 = getMinExp(r);
//	int exp2 = getMinExp(c);
//	int newRow = pow(2, exp1);
//	int newCol = pow(2, exp2);
//	int length = std::min(newRow, newCol);
//	int newLength = length;
//	while (newLength > length / 8)
//	{
//		newLength /= 2;
//	}
//	return newLength;
//}
//
//int getMinExp(int length)
//{
//	int i = 0;
//	int k = 0;
//	while (1)
//	{
//		if (length > pow(2, i) && length > pow(2, i + 1))
//		{
//			i++;
//		}
//		else {
//			if (length == pow(2, i))
//			{
//				k = i;
//			}
//			else
//			{
//				k = i + 1;
//			}
//			break;
//		}
//	}
//	return k;
//}
//
////筛选特征点
//void filtrateFeatures(cv::Mat &image)
//{
//	int newLength = drawGrid(image);
//
//}
//
//int main()
//{
//	// 读取模板图像
//	cv::Mat model = imread("Template.jpg");
//	int newLength = drawGrid(model);
//	system("pause");
//	return 0;
//}