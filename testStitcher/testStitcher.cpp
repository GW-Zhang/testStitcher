#include"myStitcher.h"

//高斯金字塔下采样
cv::Mat paramidGaussImage(cv::Mat &src, int paramidNums)
{
	int paramidCounts = paramidNums;
	cv::Mat srcPre = src;
	cv::Mat srcTemp;
	while (paramidCounts > 1)
	{
		cv::pyrDown(srcPre, srcTemp, cv::Size(srcPre.cols / 2, srcPre.rows / 2));
		srcPre = srcTemp;
		paramidCounts--;
	}
	return srcPre;
}

/*1.获取每个视角螺纹所在子图像*/

//在文件中读取图像
char * GetFileName(const char * dir, int i, int j)
{

	char *name = new char[100];
	if (i < 10) {
		sprintf(name, "%s\\000%d-00%d.jpg", dir, i, j);
		return name;
	}
	else if (i >= 10) {
		sprintf(name, "%s\\00%d-00%d.jpg", dir, i, j);
		return name;
	}


}

//将输入的图片逆时针旋转
void rotate(cv::Mat &image, cv::Mat &rectImage, int angle)
{

	//rectImage.create(image.cols, image.rows, CV_8UC1);
	//rectImage = cv::Scalar(0);
	//for (int y = 0; y < image.rows; y++)
	//{
	//	for (int x = 0; x < image.rows; x++)
	//	{
	//		rectImage.at<uchar>(x,y) = image.at<uchar>(y,x);
	//	}
	//}

	float radian = (float)(0.5 * CV_PI);

	//填充图像
	int maxBorder = (int)(max(image.cols, image.rows)* 1.414); //即为sqrt(2)*max
	int dx = (maxBorder - image.cols) / 2;
	int dy = (maxBorder - image.rows) / 2;
	copyMakeBorder(image, rectImage, dy, dy, dx, dx, BORDER_CONSTANT);
	//旋转
	Point2f center((float)(rectImage.cols / 2), (float)(rectImage.rows / 2));
	Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);//求得旋转矩阵
	warpAffine(rectImage, rectImage, affine_matrix, rectImage.size());
	//计算图像旋转之后包含图像的最大的矩形
	float sinVal = abs(sin(radian));
	float cosVal = abs(cos(radian));
	Size targetSize((int)(image.cols * cosVal + image.rows * sinVal),
		(int)(image.cols * sinVal + image.rows * cosVal));

	//剪掉多余边框
	int x = (rectImage.cols - targetSize.width) / 2;
	int y = (rectImage.rows - targetSize.height) / 2;
	Rect rect(x, y, targetSize.width, targetSize.height);
	rectImage = Mat(rectImage, rect);

}

//通过自定义框，选中螺纹区域，之后将子区域提取出来
void getWhorl(cv::Mat &image, cv::Mat &rectImage, int x, int width)
{

	//将图像改为单通道灰度图
	//cv::Mat grayImage(image.size(),image.type(),cv::Scalar(0));
	//cv::cvtColor(image, grayImage, COLOR_BGR2GRAY);

	//自定义矩形框信息，将螺纹区域的左右边界确定，切割出来


	cv::Rect rect(x, 0, width, image.rows);
	cv::Mat rectImage01 = image(rect);
	//cv::Rect rect(360, 0, 168, image.rows);
	//rectImage01 = image(rect);

	//对已经切割的图像进行二值化处理
	cv::Mat threImage;
	cv::threshold(rectImage01, threImage, 200, 255, cv::THRESH_BINARY_INV);

	////利用形态学方法对边角进行修整
	//cv::Mat erodedImage;
	//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	//cv::erode(threImage, erodedImage, element, cv::Point(-1, -1), 5);		//腐蚀

	////对其进行膨胀处理，将间隙填充
	//cv::Mat dilatedImage;
	//cv::dilate(erodedImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 1);
	//通过连通域分析，将其框起来，即确定其上下边界
	//定义轮廓数值
	std::vector<std::vector<cv::Point>>contours;
	cv::findContours(threImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	//删除无效轮廓,找出最大轮廓
	int max = contours[0].size();
	int k = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() >= max) {
			max = contours[i].size();
			k = i;
		}
	}
	//切割
	cv::Rect r0 = cv::boundingRect(contours[k]);
	int centerY = r0.height / 2 + r0.y;
	int leftBegin = centerY - 178;
	//int leftBegin = r0.tl().y;
	//cv::Rect r1(r0.x - 100 + x, r0.y, r0.width + 300, r0.height);
	cv::Rect r1(r0.x - 55 + x, leftBegin, r0.width + 55, 356);
	//cv::Rect r1(r0.x - 55 + x, leftBegin, r0.width + 55, r0.height);
	//rectImage = rectImage01(r0);
	cv::Mat rectImage02 = image(r1);
	//进行旋转调整

	rotate(rectImage02, rectImage, 270);

}

//读取图像

//拼接全部图像
void allStitcher(std::vector<cv::Mat>&vImg, cv::Mat &dst, int *arr, int cutAdjX, int *arrCutX, int width)
{
	std::vector<cv::Mat>vStitcherImg;		//用来存储拼接后的图像
	int overallWidth = 0;					//用来存放一个完整周期的宽度
	int tempWidth = 0;					//选取的模板宽度，为后续周期调整做准备
	for (int j = 0; j < 36; j++)
	{
		vImg.resize(6);
		string imagePath1, imagePath2, imagePath3;		//获取路径
		for (int i = 0; i < 2; i++)
		{
			double start0 = (double)getTickCount();
			imagePath1 = GetFileName("D:\\files\\testImage\\SaveImage-正常-5.24\\螺纹检测\\合格\\0", j, i);
			imagePath2 = GetFileName("D:\\files\\testImage\\SaveImage-正常-5.24\\螺纹检测\\合格\\1", j, i);
			imagePath3 = GetFileName("D:\\files\\testImage\\SaveImage-正常-5.24\\螺纹检测\\合格\\2", j, i);

			//1.读取图像
			cv::Mat img1 = cv::imread(imagePath1);
			cv::Mat img2 = cv::imread(imagePath2);
			cv::Mat img3 = cv::imread(imagePath3);

			double end0 = (double)getTickCount();
			double time0 = 1000 * (end0 - start0) / getTickFrequency();

			std::cout << "读取图像运行时间为：" << time0 << endl;

			if (img1.empty() || img2.empty() || img3.empty())
			{
				cout << "image open error" << endl;

			}

			double start1 = (double)getTickCount();

			//2.对读取的图像转为单通道（灰度模式）
			//寻找特征点来进行修正
			std::vector<cv::Point> vPt1;
			vPt1.resize(6);
			//detectPoints(pyrImg1, vPt1[0], vPt1[1], 200);
			//detectPoints(pyrImg2, vPt1[2], vPt1[3], 200);
			//detectPoints(pyrImg3, vPt1[4], vPt1[5], 200);
			findFeaturePoints(img1, cutAdjX, width, vPt1[0], vPt1[1]);
			findFeaturePoints(img2, cutAdjX, width, vPt1[2], vPt1[3]);
			findFeaturePoints(img3, cutAdjX, width, vPt1[4], vPt1[5]);
			////利用多线程处理
			//std::thread dpThre1(findFeaturePoints,std::ref(img1), cutAdjX, width, std::ref(vPt1[0]),std::ref(vPt1[1]));
			//dpThre1.join();
			//std::thread dpThre2(findFeaturePoints, std::ref(img2), cutAdjX, width, std::ref(vPt1[2]), std::ref(vPt1[3]));
			//dpThre2.join();
			//std::thread dpThre3(findFeaturePoints, std::ref(img3), cutAdjX, width, std::ref(vPt1[4]), std::ref(vPt1[5]));
			//dpThre3.join();

			double end1 = (double)getTickCount();
			double time1 = 1000 * (end1 - start1) / getTickFrequency();
			std::cout << "寻找特征点运行时间1为：" << time1 << endl;
			//将图像进行修正

			double start12 = (double)getTickCount();

			cv::Mat adjImg1, adjImg2, adjImg3;
			//if (vPt1[0].x == vPt1[1].x)
			//{
			//	adjImg1 = img1.clone();
			//}
			//else {
			//	adjustImg(img1, adjImg1, vPt1[0], vPt1[1]);
			//}

			//if (vPt1[2].x == vPt1[3].x)
			//{
			//	adjImg2 = img2.clone();
			//}
			//else {
			//	adjustImg(img2, adjImg2, vPt1[2], vPt1[3]);
			//}

			//if (vPt1[4].x == vPt1[5].x)
			//{
			//	adjImg3 = img3.clone();
			//}
			//else {
			//	adjustImg(img3, adjImg3, vPt1[4], vPt1[5]);
			//}
			adjustImg(img1, adjImg1, vPt1[0], vPt1[1]);
			adjustImg(img2, adjImg2, vPt1[2], vPt1[3]);
			adjustImg(img3, adjImg3, vPt1[4], vPt1[5]);
			//利用多线程处理
			//std::thread t1(adjustImg, std::ref(img1), std::ref(adjImg1), std::ref(vPt1[0]), std::ref(vPt1[1]));
			//t1.join();
			//std::thread t2(adjustImg, std::ref(img2), std::ref(adjImg2), std::ref(vPt1[2]), std::ref(vPt1[3]));
			//t2.join();
			//std::thread t3(adjustImg, std::ref(img3), std::ref(adjImg3), std::ref(vPt1[4]), std::ref(vPt1[5]));
			//t3.join();



			//释放容器
			std::vector<cv::Point>().swap(vPt1);

			double end12 = (double)getTickCount();
			double time12 = 1000 * (end12 - start12) / getTickFrequency();
			std::cout << "修正运行时间12为：" << time12 << endl;


			double start2 = (double)getTickCount();
			//3.对图像进行切割处理，切割出螺纹部分子图像
			cv::Mat rectImg1, rectImg2, rectImg3;

			//getWhorl(adjImg1, rectImg1, 330, 100);
			//getWhorl(adjImg2, rectImg2, 332, 100);
			//getWhorl(adjImg3, rectImg3, 326, 100);
			getWhorl(adjImg1, rectImg1, arrCutX[0], width);
			getWhorl(adjImg2, rectImg2, arrCutX[1], width);
			getWhorl(adjImg3, rectImg3, arrCutX[2], width);
			//std::thread RectThre1(getWhorl,std::ref(adjImg1), std::ref(rectImg1), arrCutX[0], width);
			//std::thread RectThre2(getWhorl, std::ref(adjImg2), std::ref(rectImg2), arrCutX[1], width);
			//std::thread RectThre3(getWhorl, std::ref(adjImg3), std::ref(rectImg3), arrCutX[2], width);
			//RectThre1.join();
			//RectThre2.join();
			//RectThre3.join();

			double end2 = (double)getTickCount();
			double time2 = 1000 * (end2 - start2) / getTickFrequency();
			std::cout << "切割运行时间2为：" << time2 << endl;

			//将螺纹展开图像存入容器中
			vImg[i] = rectImg1;
			vImg[i + 2] = rectImg3;
			vImg[i + 4] = rectImg2;

			//double start3 = (double)getTickCount();
			////4.对切割出的子图像以固定半径进行螺纹展开
			//int radius = 178;
			//cv::Mat spreadImg1, spreadImg2, spreadImg3;
			//speardCylinder(rectImg1, spreadImg1, radius);
			//speardCylinder(rectImg2, spreadImg2, radius);
			//speardCylinder(rectImg3, spreadImg3, radius);
			//double end3 = (double)getTickCount();
			//double time3 = 1000 * (end3 - start3) / getTickFrequency();
			//std::cout << "展开运行时间3为：" << time3*2 << endl;

			////将螺纹展开图像存入容器中
			//vImg[i] = spreadImg1;
			//vImg[i + 2] = spreadImg3;
			//vImg[i + 4] = spreadImg2;
		}
		double start3 = (double)getTickCount();
		//计算展开矩阵
		cv::Mat mapX, mapY;
		int radius;
		speardCylinderMatrix(vImg[0], radius, mapX, mapY);
		//定义容器存储展开图像
		std::vector<cv::Mat>vSpeImg;
		vSpeImg.resize(6);
		for (int i = 0; i < 6; i++)
		{
			//利用重投影矩阵，进行柱体展开
			cv::remap(vImg[i], vSpeImg[i], mapX, mapY, cv::INTER_LINEAR);
		}
		double end3 = (double)getTickCount();
		double time3 = 1000 * (end3 - start3) / getTickFrequency();
		std::cout << "展开运行时间3为：" << time3 << endl;
		//5.采用新方法（最大相关性）进行拼接
		double start4 = (double)getTickCount();
		getMostCorr(vSpeImg, dst, arr, overallWidth, tempWidth);
		double end4 = (double)getTickCount();
		double time4 = 1000 * (end4 - start4) / getTickFrequency();
		std::cout << "拼接运行时间4为：" << time4 << endl;

		////5.进行拼接
		////保存拼接后的结果
		//double start4 = (double)getTickCount();
		//cv::Mat temp;
		//getStitcherImage(vImg[0], vImg[1], temp, arr[0]);
		//double end4 = (double)getTickCount();
		//double time4 = 1000 * (end4 - start4) / getTickFrequency();
		//std::cout << "匹配拼接一次运行时间4为：" << time4 << endl;
		//for (int i = 1; i < vImg.size() - 1; ++i)
		//{

		//	getStitcherImage(temp, vImg[i + 1], dst, arr[i]);
		//	temp = dst;
		//}

		//6.进行周期切割，使其都保证首尾一致
		int dis = 0;
		cv::Point p;
		getPeriod(vImg[0], vImg[5], 60, dis,p);
		int distance = dst.cols - dis-p.x+20;			//确定切割的长度
		cv::Mat resultImg = dst(cv::Rect(p.x,0,distance,dst.rows));

		vStitcherImg.push_back(dst);

		cv::imshow(GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹A检测\\合格\\0", j, 0), resultImg);
	}
	//释放容器空间
	std::vector<cv::Mat>().swap(vImg);
	//7.调整拼接后的图像周期，使其螺纹区域在同一周期

	cv::Mat target = vStitcherImg[0](cv::Rect(0, 30, tempWidth, vStitcherImg[0].rows - 55));

	for (int i = 1; i < vStitcherImg.size(); i++)
	{
		cv::Mat SamePeriodImg;
		adjPeriod(target, vStitcherImg[i], overallWidth, SamePeriodImg);
		//cv::imshow(" ", SamePeriodImg);
		cv::imshow(GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹A检测\\合格\\0", i,0), SamePeriodImg);
		//cv::waitKey(0);
	}


}
//对图像进行处理
//1.切割
//2.展开
//3.拼接
//void getImg(std::vector<cv::Mat>&vImg,int *arr, cv::Mat &dst)
//{
//	//1.获取螺纹图像
//	//getImg(vImg);
//
//	//2.进行拼接
//			//保存拼接后的结果
//	double start4 = (double)getTickCount();
//	cv::Mat temp ;
//	getStitcherImage(vImg[0], vImg[1], temp, arr[0]);
//	double end4 = (double)getTickCount();
//	double time4 = 1000 * (end4 - start4) / getTickFrequency();
//	cout << "匹配拼接一次运行时间4为：" << time4 << endl;
//	for (int i =1; i < vImg.size()-1; ++i)
//	{
//		
//		getStitcherImage(temp, vImg[i + 1], dst, arr[i]);
//		temp = dst;
//	}
//	cv::imshow("01", dst);
//	
//}

//读入图像，并对一个相机得两幅图像进行拼接
//void allWhorl(int *arr01, std::vector<cv::Mat>&vAllimg)
//{
//
//	string imagePath, imagePath1, imagePath2;
//
//	std::vector<cv::Mat>vImg, vImg1, vImg2;
//
//	//读取图像
//	for (int i = 0; i < 2; i++) {
//		imagePath = GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹A检测\\合格\\0", i,0);
//		imagePath1 = GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹B检测\\合格\\0", i,0);
//		imagePath2 = GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹C检测\\合格\\0", i,0);
//		cv::Mat img = cv::imread(imagePath);
//		cv::Mat img1 = cv::imread(imagePath1);
//		cv::Mat img2 = cv::imread(imagePath2);
//		if (img.empty() || img1.empty() || img2.empty())
//		{
//			std::cout << "image open error" << endl;
//
//		}
//
//		//进行灰度处理，转为单通道
//		cv::Mat gray, gray1, gray2;
//		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
//		cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
//		cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
//
//		//进行螺纹子区域的切割
//
//		vImg.push_back(gray);
//		vImg1.push_back(gray1);
//		vImg2.push_back(gray2);
//	}
//
//
//
//	cv::Mat rectImg, rectImg1, rectImg2;
//	cv::Mat speImg, speImg1, speImg2;
//	std::vector<cv::Mat>vSpeImg;
//	vSpeImg.resize(vImg.size());
//	std::vector<cv::Mat>vSpeImg1;
//	vSpeImg1.resize(vImg1.size());
//	std::vector<cv::Mat>vSpeImg2;
//	vSpeImg2.resize(vImg2.size());
//
//	std::vector<cv::Mat>pyrSpeImg, pyrSpeImg1, pyrSpeImg2;
//
//	int radius = 178;
//	for (int i = 0; i < vImg.size(); i++)
//	{
//		//1.切割子图像
//		getWhorl(vImg[i], rectImg, 340, 105);
//		//2.螺纹展开
//		speardCylinder(rectImg, vSpeImg[i], radius);
//		//vSpeImg.push_back(speImg);
//		//pyrSpeImg.push_back(paramidGaussImage(vSpeImg[i], 2));
//
//		getWhorl(vImg1[i], rectImg1, 357, 105);
//		speardCylinder(rectImg1, vSpeImg1[i], radius);
//		//vSpeImg1.push_back(speImg1);
//		//pyrSpeImg1.push_back(paramidGaussImage(vSpeImg1[i], 2));
//
//		getWhorl(vImg2[i], rectImg2, 365, 105);
//		speardCylinder(rectImg2, vSpeImg2[i], radius);
//		//vSpeImg2.push_back(speImg2);
//		//pyrSpeImg2.push_back(paramidGaussImage(vSpeImg2[i], 2));
//	}
//
//
//	//释放容器空间
//	std::vector<cv::Mat>().swap(vImg);
//	std::vector<cv::Mat>().swap(vImg1);
//	std::vector<cv::Mat>().swap(vImg2);
//
//	//先进行一副相机的两张图像进行拼接
//
//	//cv::Mat dst,dst1,dst2;
//	//for (int i = 0; i < vSpeImg.size() - 1; i += 2)
//	//{
//	//	//StitcherImg(pyrSpeImg1[i], pyrSpeImg1[i+1],vSpeImg1[i], vSpeImg1[i + 1], dst1, arr01[0]);
//	//	getStitcherImage(vSpeImg1[i], vSpeImg1[i + 1], dst1, arr01[0]);
//	//	vAllimg.push_back(dst1);
//
//	//	//StitcherImg(pyrSpeImg[i], pyrSpeImg[i + 1], vSpeImg[i], vSpeImg[i + 1], dst, arr01[1]);
//	//	getStitcherImage(vSpeImg[i], vSpeImg[i + 1], dst, arr01[1]);
//	//	vAllimg.push_back(dst);
//
//	//	//StitcherImg(pyrSpeImg2[i], pyrSpeImg2[i + 1], vSpeImg2[i], vSpeImg2[i + 1], dst2, arr01[2]);
//	//	getStitcherImage(vSpeImg2[i], vSpeImg2[i + 1], dst2, arr01[2]);
//	//	vAllimg.push_back(dst2);
//
//	//}
//	cv::Mat dst, dst1, dst2;
//	for (int i = 0; i < vSpeImg.size() - 1; i += 2)
//	{
//
//		getStitcherImage(vSpeImg1[i], vSpeImg1[i + 1], dst1, arr01[0]);
//		vAllimg.push_back(dst1);
//
//		getStitcherImage(vSpeImg[i], vSpeImg[i + 1], dst, arr01[1]);
//		vAllimg.push_back(dst);
//
//		getStitcherImage(vSpeImg2[i], vSpeImg2[i + 1], dst2, arr01[2]);
//		vAllimg.push_back(dst2);
//
//	}
//	//释放容器空间
//	std::vector<cv::Mat>().swap(vSpeImg);
//	std::vector<cv::Mat>().swap(vSpeImg1);
//	std::vector<cv::Mat>().swap(vSpeImg2);
//
//	//for (int i = 0; i < vAllimg.size(); i++)
//	//{
//	//	cv::imshow(" ", vAllimg[i]);
//	//	waitKey(1200);
//
//	//}
//
//
//}

/*2.获取螺旋区域直径，进行柱面展开*/

void speardCylinderMatrix(cv::Mat &rectImage01, int &radius, cv::Mat &mapX, cv::Mat &mapY)
{

	radius = rectImage01.cols / 2.0;		//定义半径，为子图像宽的一半
	cv::Mat speImage(rectImage01.rows, PI*radius, CV_8UC1);

	//映射参数
	mapX.create(speImage.size(), CV_32F);
	mapY.create(speImage.size(), CV_32F);

	//创建映射参数
	for (int y = 0; y < speImage.rows; y++)
	{
		for (int x = 0; x < speImage.cols; x++)
		{

			mapY.at<float>(y, x) = (y);
			mapX.at<float>(y, x) = (radius*sin(((x - speImage.cols / 2.0)) / radius) + radius);

		}
	}
	//cv::remap(rectImage, speImage, mapX, mapY, cv::INTER_LINEAR);
	//	cv::imshow("重投影", speImage);
}

/*3. 根据相邻两个区域之间夹角的初始值，在展开图重叠部分进行模板匹配，
在瓶胚的竖直方向上匹配搜索的范围可以小一点，然后将同一相机的图像进行拼接；*/

/*3.1 通过模板匹配的方法求取平移变换参数*/
/**求平移量
 *参数表为输入两幅图像（有一定重叠区域）
 *返回值为点类型，存储x,y方向的偏移量
*/


//利用模板匹配，进行图像拼接
void getStitcherImage(cv::Mat &speImage01, cv::Mat &speImage02, cv::Mat& dstImg, int angle)
{

	int c = speImage02.cols;	//为单张展开图像的大约宽度
	int col = (c / 2 + (c*double(angle / 180.0) / 2));
	int Rcol = c - col;

	cv::Rect rectCut(col, 40, 50, speImage02.rows - 45);			//定义模板位置大小
	cv::Rect rectMatched(c / 4, 0, c / 4 + 50, speImage01.rows);		//定义匹配位置
	cv::Mat imgTemp = speImage02(rectCut);			//在左图像上取模板
	cv::Mat imgMatched = speImage01(rectMatched);			//取右图像左半区域进行模板匹配

	int width = imgMatched.cols - imgTemp.cols + 1;
	int height = imgMatched.rows - imgTemp.rows + 1;
	cv::Mat matchResult(height, width, CV_32FC1);
	cv::matchTemplate(imgMatched, imgTemp, matchResult, cv::TM_CCORR_NORMED);			//匹配
	cv::normalize(matchResult, matchResult, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围

	double minValue, maxValue;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(matchResult, &minValue, &maxValue, &minLoc, &maxLoc);		//找到最相似的位置
	std::cout << "匹配得分为：" << maxValue << endl;

	//定义拼接图像
	int newCol = maxLoc.x + c / 4;
	int newRow = maxLoc.y - 40;			//y轴方向的偏移度
	cv::Mat dst(speImage01.rows + abs(newRow), speImage01.cols + rectCut.x - newCol, CV_8UC1);
	if (newRow <= 0) {
		cv::Mat roiLeft = dst(Rect(0, 0, speImage02.cols, speImage02.rows));	//公共区域左部分
		speImage02.copyTo(roiLeft);
	}
	else {
		cv::Mat roiLeft = dst(Rect(0, newRow, speImage02.cols, speImage02.rows));	//公共区域左部分
		speImage02.copyTo(roiLeft);
	}


	////在有图上画出模板区域

	//cv::Mat rDebugImg = speImage01.clone();
	//cv::rectangle(rDebugImg, Rect(newCol, 0, imgTemp.cols, imgTemp.rows), Scalar(255, 255, 0), 2, 8);
	////在左图上画出模板区域
	//cv::Mat lDebugImg = speImage02.clone();
	//cv::rectangle(lDebugImg, Rect(col, 0, imgTemp.cols, imgTemp.rows), Scalar(255, 255, 0), 2, 8);

	//拼接公共区域右半部分
	cv::Mat roiMatched = speImage01(Rect(newCol, 0, speImage01.cols - newCol, speImage01.rows - 1));
	if (newRow >= 0) {
		cv::Mat roiRight = dst(Rect(rectCut.x, 0, roiMatched.cols, roiMatched.rows));
		roiMatched.copyTo(roiRight);
	}
	else {
		cv::Mat roiRight = dst(Rect(rectCut.x, abs(newRow), roiMatched.cols, roiMatched.rows));
		roiMatched.copyTo(roiRight);
	}


	//利用加权，进行图像融合处理(公共区域宽度为20)

	cv::resize(imgTemp, imgTemp, cv::Size(20, imgTemp.rows));
	//cv::Mat leftTempImg = speImage02(cv::Rect(rectCut));//imgTemp;
	cv::Mat leftTempImg = speImage02(cv::Rect(rectCut.x, rectCut.y, 20, imgTemp.rows));
	cv::Mat rightTempImg = speImage01(Rect(newCol, maxLoc.y, 20, imgTemp.rows));
	cv::Mat mergeImg(imgTemp.size(), imgTemp.type(), cv::Scalar(0));		//融合图像
	for (int y = 0; y < imgTemp.rows; y++)
	{
		for (int x = 0; x < 20; x++)
		{
			int leftPixel = leftTempImg.at<uchar>(y, x);
			int rightPixel = rightTempImg.at<uchar>(y, x);
			mergeImg.at<uchar>(y, x) = (1 - (0.05*x))*leftPixel
				+ (0.05*x)*rightPixel;

		}

	}
	//cv::addWeighted(leftTempImg, 0.5, rightTempImg, 0.5, 0, mergeImg);
	if (newRow <= 0) {
		cv::Mat roi = dst(Rect(rectCut.x, 40, imgTemp.cols, imgTemp.rows));
		mergeImg.copyTo(roi);
	}
	else {
		cv::Mat roi = dst(Rect(rectCut.x, 40 + newRow, imgTemp.cols, imgTemp.rows));
		mergeImg.copyTo(roi);
	}



	dstImg.create(dst.size(), dst.type());
	dstImg = dst.clone();
	//	cv::imshow("融合之后", dstImg);


}

//多张图像进行拼接融合
void mulStitcherImage(std::vector<cv::Mat>&vSpeImg, cv::Mat &dst, int num, int *arr)
{
	cv::Mat temp = vSpeImg[0];

	//getStitcherImage(vSpeImg[0], vSpeImg[1], temp, angle);
	cv::Mat sticher;
	//std::vector<cv::Mat>vSticher;
	//vSticher.resize(num);

	for (int i = 1; i < num; i++)
	{

		getStitcherImage(temp, vSpeImg[i], sticher, arr[i - 1]);
		//temp.create(dst.size(), dst.type());
		temp = sticher;


	}
	dst = temp.clone();
}



//做一个首尾切割函数，确定周期，对拼接图像进行首尾切割，使其在一个周期
void getPeriod(cv::Mat &img01, cv::Mat &img02, int angle, int &dis, cv::Point &p)
{
	//首先根据旋转角度，确定近似匹配位置
	//在匹配左图的匹配位置列为
	int c = img01.cols;	//为单张展开图像的大约宽度
	//利用旋转角度，确定大概的拼接位置

	int col = (c / 2 + (c*double(angle / 180.0) / 2));
	//int col= (img01.cols / 2 + (img01.cols *double(angle / 180.0) / 2));
	dis = img01.cols - col;			//此列到尾列的距离

	//定义模板尺寸以及匹配位置
	cv::Mat tempImg = img01(Rect(col, 20, 50, img01.rows - 25));
	cv::Mat matchImg = img02(Rect(c / 4, 0, c / 3, img02.rows));

	//模板匹配
	cv::Mat result(matchImg.rows - tempImg.rows + 1, matchImg.cols - tempImg.cols + 1, CV_32FC1);
	cv::matchTemplate(matchImg, tempImg, result, cv::TM_CCORR_NORMED);
	cv::normalize(result, result, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围

	//找到最相似的位置
	double minVal, maxVal;
	cv::Point minPt, maxPt;
	cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);
	p.x = maxPt.x + c / 4;

	////在有图上画出模板区域

	//cv::Mat rDebugImg = img02.clone();
	//cv::rectangle(rDebugImg, Rect(maxPt.x, maxPt.y, tempImg.cols, tempImg.rows), Scalar(255, 255, 0), 2, 8);
	////在左图上画出模板区域
	//cv::Mat lDebugImg = img01.clone();
	//cv::rectangle(lDebugImg, Rect(col, 0, tempImg.cols, tempImg.rows), Scalar(255, 255, 0), 2, 8);


}

//对所有函数进行整合
void resultImg(std::vector<cv::Mat>&vAllimg, int num, int *arr)
{

	std::vector<cv::Mat>vResult;
	for (int j = 0; j < vAllimg.size(); j += 3)
	{
		//	//std::vector<cv::Mat>vRectImg;
		////vRectImg.resize(num);
		//	std::vector<cv::Mat>vSpeImg;
		//	cv::Mat speImg;
		//	for (int i = 0; i < num; i++)
		//	{
		//		//1.切割子图像
		//		//getWhorl(vAllimg[i], vRectImg[i], 360, 168);
		//	//	writeImg(GetFileName("D:\\files\\testImage", i), vRectImg[i]);
		//		//cv::imshow(GetFileName("D:\\files\\testImage", i), vRectImg[i]);
		//		//2.将切割出来的子图像进行柱体展开
		//		speardCylinder(vAllimg[i+j], speImg);
		//		vSpeImg.push_back(speImg);
		//		//cv::imwrite(GetFileName("D:\\files\\testImage", i), vSpeImg[i]);

		//	}
			//3.拼接
		cv::Mat dst;
		cv::Mat temp = vAllimg[j];

		//getStitcherImage(vSpeImg[0], vSpeImg[1], temp, angle);
		cv::Mat sticher;
		for (int i = 1; i < num; i++)
		{
			getStitcherImage(temp, vAllimg[i + j], sticher, arr[i - 1]);
			//temp.create(dst.size(), dst.type());
			temp = sticher;
		}
		dst = temp.clone();
		cv::imshow(GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹A检测\\合格\\0", j, 0), dst);
		//mulStitcherImage(vAllimg, dst, num, arr);
		//	cv::imshow("拼接图像", dst);
			//4.确定周期，对拼接图像进行首尾切割，使其在一个周期
		cv::Point p;
		int distance;
		getPeriod(vAllimg[j], vAllimg[j + 2], 70, distance, p);
		int weight = dst.cols - p.x - distance + 20;
		int height = dst.rows;
		cv::Mat result = dst(Rect(p.x, 0, weight, height));
		vResult.push_back(result);
		//cv::imshow("结果图像", result);
	}

	//释放容器空间
	std::vector<cv::Mat>().swap(vAllimg);
	std::vector<cv::Mat>().swap(vResult);
}




//将所有拼接图像进行调整到相同周期下（开始、结束相同）
void adjPeriod(cv::Mat &target, cv::Mat &img, int overallWidth, cv::Mat &adjImg)
{

	int weight = img.cols - target.cols + 1;
	int height = img.rows - target.rows + 1;
	cv::Mat result(height, weight, CV_32FC1);

	//进行模板匹配
	cv::matchTemplate(img, target, result, cv::TM_CCORR_NORMED);
	cv::normalize(result, result, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围
	//找到最相似位置
	double minVal, maxVal;
	cv::Point minPt, maxPt;
	cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);

	//对图像周期进行调整，将其进行切割拼接
	//int newCol = 30 - maxPt.y;
	//adjImg.create(img.rows+abs(newCol), overallWidth, CV_8UC1);
	//if (newCol < 0) {
	//	cv::Mat roiLeft = adjImg(Rect(0, 0, adjImg.cols - maxPt.x - 1, img.rows));
	//	img(Rect(maxPt.x, 0, adjImg.cols - maxPt.x - 1, adjImg.rows)).copyTo(roiLeft);

	//	cv::Mat roiRight = adjImg(Rect(adjImg.cols - maxPt.x - 1, abs(newCol), maxPt.x + 1, adjImg.rows));
	//	img(Rect(0, 0, maxPt.x + 1, img.rows)).copyTo(roiRight);
	//}
	//else {
	//	cv::Mat roiLeft = adjImg(Rect(0,newCol, adjImg.cols - maxPt.x - 1, img.rows));
	//	img(Rect(maxPt.x, 0, adjImg.cols - maxPt.x - 1, adjImg.rows)).copyTo(roiLeft);

	//	cv::Mat roiRight = adjImg(Rect(adjImg.cols - maxPt.x - 1, 0, maxPt.x + 1, adjImg.rows));
	//	img(Rect(0, 0, maxPt.x + 1, img.rows)).copyTo(roiRight);
	//}
	adjImg.create(img.rows + 0, overallWidth, CV_8UC1);
	cv::Mat roiLeft = adjImg(Rect(0, 0, adjImg.cols - maxPt.x - 1, img.rows));
	img(Rect(maxPt.x, 0, adjImg.cols - maxPt.x - 1, adjImg.rows)).copyTo(roiLeft);

	cv::Mat roiRight = adjImg(Rect(adjImg.cols - maxPt.x - 1, 0, maxPt.x + 1, adjImg.rows));
	img(Rect(0, 0, maxPt.x + 1, img.rows)).copyTo(roiRight);

	////对拼接部分进行融合
	//cv::Mat leftEnd = img(Rect(img.cols - 20 - 1, 0, 20, adjImg.rows));
	//cv::Mat rightHead = img(Rect(0, 0, 20, adjImg.rows));
	//cv::Mat middle(leftEnd.size(), leftEnd.type(), cv::Scalar(0));
	//for (int y = 0; y < adjImg.rows; y++)
	//{
	//	for (int x = 0; x < 20; x++)
	//	{
	//		int leftPixel = leftEnd.at<uchar>(y, x);
	//		int rightPixel = rightHead.at<uchar>(y, x);
	//		middle.at<uchar>(y, x) = (1 - (x * 0.05))*leftPixel
	//			+ (x*0.05) * rightPixel;
	//	}
	//}
	//middle.copyTo(adjImg(Rect(adjImg.cols - maxPt.x - 1, 0, 20, adjImg.rows)));


}

//获取拼接位置
void getPyrMatch(cv::Mat &speImage01, cv::Mat &speImage02, int angle, cv::Point pt, cv::Point maxLoc)
{

	int c = speImage02.cols;	//为单张展开图像的大约宽度
	//利用旋转角度，确定大概的拼接位置
	//int col = speImage02.cols - 280 - (c*double(angle / 180.0) / 2);
	//int col = (280+(c*double(angle/180.0)/2));
	int col = (c / 2 + (c*double(angle / 180.0) / 2));
	pt.x = col;
	pt.y = 0;

	cv::Rect rectCut(col, 20, 50, speImage02.rows - 25);			//定义模板位置大小
	cv::Rect rectMatched(c / 4, 0, c / 3, speImage01.rows);
	cv::Mat imgTemp = speImage02(rectCut);			//在左图像上取模板
	cv::Mat imgMatched = speImage01(rectMatched);			//取右图像左半区域进行模板匹配

	int width = imgMatched.cols - imgTemp.cols + 1;
	int height = imgMatched.rows - imgTemp.rows + 1;
	cv::Mat matchResult(height, width, CV_32FC1);
	cv::matchTemplate(imgMatched, imgTemp, matchResult, cv::TM_CCORR_NORMED);			//匹配
	cv::normalize(matchResult, matchResult, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围

	double minValue, maxValue;
	cv::Point minLoc;
	cv::minMaxLoc(matchResult, &minValue, &maxValue, &minLoc, &maxLoc);		//找到最相似的位置



}

//对高斯金字塔处理过后的匹配图像进行拼接
void StitcherImg(cv::Mat &pyrSpeImg01, cv::Mat &pyrSpeImg02, cv::Mat &speImage01, cv::Mat &speImage02, cv::Mat& dstImg, int angle)
{
	cv::Point leftPt, rightPt;
	getPyrMatch(speImage01, speImage02, angle, leftPt, rightPt);
	cv::Point maxLoc;
	if (speImage01.cols / 2 == 0) {
		maxLoc.x = 2 * rightPt.x;
	}
	else {
		maxLoc.x = 2 * rightPt.x + 1;
	}
	//定义拼接图像
	int c = 500;	//为单张展开图像的大约宽度
	int newCol = maxLoc.x + c / 5;
	int col = 0;
	if (speImage02.cols / 2 == 0) {
		col = 2 * leftPt.x;
	}
	cv::Rect rectCut(col, 0, 20, speImage02.rows);			//定义模板位置大小
	cv::Rect rectMatched(c / 5, 0, c / 3, speImage01.rows);
	cv::Mat imgTemp = speImage02(rectCut);			//在左图像上取模板
	cv::Mat imgMatched = speImage01(rectMatched);			//取右图像左半区域进行模板匹配
	cv::Mat dst(speImage01.rows, speImage01.cols + col - newCol, CV_8UC1);
	cv::Mat roiLeft = dst(Rect(0, 0, col, speImage02.rows));	//公共区域左部分
	speImage02.copyTo(roiLeft);

	//在有图上画出模板区域

	cv::Mat rDebugImg = speImage01.clone();
	cv::rectangle(rDebugImg, Rect(newCol, maxLoc.y, imgTemp.cols, imgTemp.rows), Scalar(255, 255, 0), 2, 8);
	//在左图上画出模板区域
	cv::Mat lDebugImg = speImage02.clone();
	cv::rectangle(lDebugImg, Rect(col, 0, imgTemp.cols, imgTemp.rows), Scalar(255, 255, 0), 2, 8);

	//拼接公共区域右半部分
	cv::Mat roiMatched = speImage01(Rect(newCol, maxLoc.y - rectCut.y, speImage01.cols - newCol, speImage01.rows - 1 - (maxLoc.y - rectCut.y)));
	cv::Mat roiRight = dst(Rect(rectCut.x, 0, roiMatched.cols, roiMatched.rows));
	roiMatched.copyTo(roiRight);

	//利用加权，进行图像融合处理
	cv::Mat leftTempImg = imgTemp;
	cv::Mat rightTempImg = speImage01(Rect(newCol, maxLoc.y, imgTemp.cols, imgTemp.rows));
	cv::Mat mergeImg(imgTemp.size(), imgTemp.type(), cv::Scalar(0));		//融合图像
	for (int y = 0; y < imgTemp.rows; y++)
	{
		for (int x = 0; x < 20; x++)
		{
			int leftPixel = leftTempImg.at<uchar>(y, x);
			int rightPixel = rightTempImg.at<uchar>(y, x);
			mergeImg.at<uchar>(y, x) = (1 - (0.05*x))*leftPixel
				+ (0.05*x)*rightPixel;

		}

	}
	//cv::addWeighted(leftTempImg, 0.5, rightTempImg, 0.5, 0, mergeImg);

	cv::Mat roi = dst(Rect(rectCut.x, 0, imgTemp.cols, imgTemp.rows));
	mergeImg.copyTo(roi);


	dstImg.create(dst.size(), dst.type());
	dstImg = dst.clone();
	//	cv::imshow("融合之后", dstImg);

}

//首先检测出图像的特征点
void detectPoints(cv::Mat &img, cv::Point &pt1, cv::Point &pt2, int X)
{
	//转为单通道图像
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	//首先切割图像，去除吸胚部分的影响
	cv::Rect rect(X, 0, 100, img.rows);
	cv::Mat img1 = gray(rect);

	//对图像进行滤波操作，降低噪声
	cv::Mat BlurImg;
	cv::GaussianBlur(img1, BlurImg, cv::Size(5, 5), 3, 3);

	//对切割的图像进行二值化处理
	cv::Mat threImg;
	cv::threshold(BlurImg, threImg, 150, 255, cv::THRESH_BINARY_INV);

	//利用投影方法将像素累积到边界,以此来计算特征
	int arr[1500] = { 0 };
	for (int y = 0; y < threImg.rows; ++y)
	{
		int num = 0;
		for (int x = 0; x < threImg.cols; ++x)
		{
			if (threImg.at<uchar>(y, x) != 0)
			{
				++num;
			}
		}
		arr[y] = num;
	}
	//for (int i = 0; i < threImg.rows; i++)
	//{
	//	cout << arr[i] << endl;
	//}

	//查找所有产生突变的点的位置
	//建立一个容器存储位置信息
	std::vector<int>v;
	for (int i = 0; i < threImg.rows; i++)
	{
		if ((arr[i] == 0) && (arr[i + 1] != 0))
		{
			v.push_back(i);
		}
		if ((arr[i] != 0) && (arr[i + 1] == 0))
		{
			v.push_back(i + 1);
		}
	}
	//cout << v.size() << endl;

	//通过两个点的中心点位置的像素值，判断两个点是否符合情况
	int y[100] = { 0 };
	if (arr[int((v[0] + v[1]) / 2)] != 0) {
		y[0] = v[0];
		y[1] = v[1];
	}
	else {
		y[0] = v[1];
		y[1] = v[2];
	}
	//cout << threImg.channels() << endl;
	//进行像素点定位
//	cv::Point pt1, pt2;
	for (int x = threImg.cols - 1; x >= 0; x--)
	{
		if (threImg.at<uchar>(y[0] + 5, x) != 0)
		{
			pt1.x = x + X;
			pt1.y = y[0];
			break;
		}

	}
	for (int x = threImg.cols - 1; x >= 0; x--) {
		if (threImg.at<uchar>(y[1] - 5, x) != 0)
		{
			pt2.x = x + X;
			pt2.y = y[1];
			break;
		}

	}

	//释放容器空间
	std::vector<int>().swap(v);
}

//对图像进行角度修正
void adjustImg(cv::Mat &img, cv::Mat &adjImg, cv::Point &pt1, cv::Point &pt2)
{

	//转为单通道图像
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	////首先切割图像，去除吸胚部分的影响
	//cv::Rect rect(X, 0, X+200, img.rows);
	//cv::Mat img1 = gray(rect);

	////对图像进行滤波操作，降低噪声
	//cv::Mat BlurImg;
	//cv::GaussianBlur(img1, BlurImg, cv::Size(5, 5), 3, 3);

	////对切割的图像进行二值化处理
	//cv::Mat threImg;
	//cv::threshold(BlurImg, threImg, 150, 255, cv::THRESH_BINARY_INV);

	////利用投影方法将像素累积到边界,以此来计算特征
	//int arr[1500] = { 0 };
	//for (int y = 0; y < threImg.rows; ++y)
	//{
	//	int num = 0;
	//	for (int x = 0; x < threImg.cols; ++x)
	//	{
	//		if (threImg.at<uchar>(y, x) !=0)
	//		{
	//			++num;
	//		}
	//	}
	//	arr[y] = num;
	//}
	////for (int i = 0; i < threImg.rows; i++)
	////{
	////	cout << arr[i] << endl;
	////}

	////查找所有产生突变的点的位置
	////建立一个容器存储位置信息
	//std::vector<int>v;
	//for (int i = 0; i < threImg.rows-10; i++)
	//{
	//	if ((arr[i] == 0 )&& (arr[i + 1] != 0 ))
	//	{
	//		v.push_back(i);
	//	}
	//	if ((arr[i] != 0) &&( arr[i + 1] == 0 ))
	//	{
	//		v.push_back(i+1);
	//	}
	//}
	////cout << v.size() << endl;

	//int y[100] = { 0 };
	//if (arr[int((v[0] + v[1] ) / 2)] != 0) {
	//	y[0] = v[0];
	//	y[1] = v[1];
	//}
	//else {
	//	y[0] = v[1];
	//	y[1] = v[2];
	//}
	////cout << threImg.channels() << endl;
	////进行像素点定位
	//cv::Point pt1, pt2;
	//for (int x = threImg.cols - 1; x >= 0; x--)
	//{
	//	if (threImg.at<uchar>(y[0]+5,x) != 0)
	//	{
	//		pt1.x = x+ X;
	//		pt1.y = y[0];	
	//		break;
	//	}
	//	
	//}
	//for (int x = threImg.cols-1; x >=0; x--){
	//	if (threImg.at<uchar>(y[1]-5,x) != 0)
	//	{
	//		pt2.x = x+ X;
	//		pt2.y = y[1];
	//		break;
	//	}
	//	
	//}

	//在原图中画出两点，并进行连线
	//cv::line(img, pt1, pt2, cv::Scalar(255, 0, 0), 2, 8);
	//cv::imshow(" ", gray);
	//cv::waitKey(0);
	////计算两点的旋转角度,进行旋转校正
	//double angle = double(atan((pt1.y-pt2.y)/(pt1.x-pt2.x)));
	////rotate(img, adjImg, -angle);
	//Point center(pt1.x, pt1.y);
	//Mat affine_matrix = getRotationMatrix2D(center, -angle/2.0, 1.0);//求得旋转矩阵
	//warpAffine(img, adjImg, affine_matrix, img.size());

	//计算两点x方向的偏移，进行平移校正
	gray.copyTo(adjImg);
	cv::Point center(((pt1.x + pt2.x) / 2), ((pt1.y + pt2.y) / 2));		//两点连线的中心位置
	double length = (pt2.y - pt1.y) / 2;

	double num = pt1.x - pt2.x;			//差的像素数量

	double moveNum = num / double(pt2.y - pt1.y);
	//处理中心点的上半部分
	for (int y = pt1.y; y < center.y; ++y)
	{
		double move = double(center.y - y)*moveNum;
		int move1 = int(move);
		double weight = abs(move - move1);
		for (int x = 300; x < 600; ++x)
		{

			if (num > 0) {
				int value1 = (1.0 - weight) * (gray.at<uchar>(y, x + move1));
				int value2 = (weight)*(gray.at<uchar>(y, x + move1 + 1));
				adjImg.at<uchar>(y, x) = value1 + value2;
			}
			else {
				int value1 = (1.0 - weight) * (gray.at<uchar>(y, x + move1));
				int value2 = (weight)*(gray.at<uchar>(y, x + move1 - 1));
				adjImg.at<uchar>(y, x) = value1 + value2;
			}
		}
	}

	//处理中心点的下半部分
	for (int y = center.y; y < pt2.y; ++y)
	{
		double move = double(y - center.y)*moveNum;
		int move1 = int(move);
		double weight = abs(move - move1);
		for (int x = 300; x < 600; ++x)
		{
			if (num > 0) {
				int value1 = (1.0 - weight) * (gray.at<uchar>(y, x - move1));
				int value2 = (weight)*(gray.at<uchar>(y, x - move1 - 1));
				adjImg.at<uchar>(y, x) = value1 + value2;
			}
			else {
				int value1 = (1.0 - weight) * (gray.at<uchar>(y, x - move1));
				int value2 = (weight)*(gray.at<uchar>(y, x - move1 + 1));
				adjImg.at<uchar>(y, x) = value1 + value2;
			}
		}
	}
	//cv::imshow(" ", adjImg);

	//for (int i = 0; i < abs(num)+1; i++)
	//{
	//	int length1 = (i*length) / (abs(num) + 1);
	//	int length2 = (((i + 1)*length) / (abs(num) + 1));
	//	for (int y = pt1.y + length1; y < pt1.y + length2; y++)
	//	{
	//		for (int x = 300; x < 600; x++)
	//		{
	//			if (num > 0) {
	//				adjImg.at<uchar>(y, x - (num - i)) = gray.at<uchar>(y, x);
	//			}
	//			else {
	//				adjImg.at<uchar>(y, x - (i + num)) = gray.at<uchar>(y, x);
	//			}

	//		}
	//	}
	//}
	////处理中心点的下半部分
	//for (int i = 0; i < abs(num)+1; i++)
	//{
	//	int length1 = (i*length) / (abs(num) + 1);
	//	int length2 = (((i + 1)*length) / (abs(num) + 1));
	//	for (int y = center.y + length1; y < center.y + length2; y++)
	//	{
	//		for (int x = 300; x < 600; x++)
	//		{
	//			if (num > 0) {
	//				adjImg.at<uchar>(y, x +i) = gray.at<uchar>(y, x);
	//			}
	//			else {
	//				adjImg.at<uchar>(y, x - i) =gray.at<uchar>(y, x);
	//			}

	//		}
	//	}
	//}



}

//尝试使用多模板进行匹配
void mulTempMatch(cv::Mat &speImage01, cv::Mat &speImage02, cv::Mat& dstImg, int angle)
{
	int c = speImage02.cols;	//为单张展开图像的大约宽度
	int col1 = (c / 2 + (c*double(angle / 180.0) / 2));
	int col2 = (c / 2 + (c*double((angle + 10) / 180.0) / 2));
	int col3 = (c / 2 + (c*double((angle + 20) / 180.0) / 2));

	cv::Rect rectCut1(col1, 40, 50, speImage02.rows - 45);			//定义模板位置大小
	cv::Rect rectCut2(col2, 40, 50, speImage02.rows - 45);
	cv::Rect rectCut3(col3, 40, 50, speImage02.rows - 45);
	cv::Rect rectMatched(c / 4, 0, c / 3, speImage01.rows);		//定义匹配位置
	cv::Mat imgTemp1 = speImage02(rectCut1);			//在左图像上取模板
	cv::Mat imgTemp2 = speImage02(rectCut2);
	cv::Mat imgTemp3 = speImage02(rectCut3);
	cv::Mat imgMatched = speImage01(rectMatched);			//取右图像左半区域进行模板匹配

	int width = imgMatched.cols - imgTemp1.cols + 1;
	int height = imgMatched.rows - imgTemp1.rows + 1;
	cv::Mat matchResult(height, width, CV_32FC1);

	cv::matchTemplate(imgMatched, imgTemp1, matchResult, cv::TM_CCORR_NORMED);			//匹配
	cv::normalize(matchResult, matchResult, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围
	double minValue1, maxValue1;
	cv::Point minLoc1, maxLoc1;
	cv::minMaxLoc(matchResult, &minValue1, &maxValue1, &minLoc1, &maxLoc1);		//找到最相似的位置

	cv::matchTemplate(imgMatched, imgTemp2, matchResult, cv::TM_CCORR_NORMED);			//匹配
	cv::normalize(matchResult, matchResult, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围
	double minValue2, maxValue2;
	cv::Point minLoc2, maxLoc2;
	cv::minMaxLoc(matchResult, &minValue2, &maxValue2, &minLoc2, &maxLoc2);		//找到最相似的位置

	cv::matchTemplate(imgMatched, imgTemp3, matchResult, cv::TM_CCORR_NORMED);			//匹配
	cv::normalize(matchResult, matchResult, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围
	double minValue3, maxValue3;
	cv::Point minLoc3, maxLoc3;
	cv::minMaxLoc(matchResult, &minValue3, &maxValue3, &minLoc3, &maxLoc3);		//找到最相似的位置

	int dis1 = col2 - col1;
	int dis2 = col3 - col2;
	int dis3 = col3 - col1;

	int matchDis1 = maxLoc2.x - maxLoc1.x;
	int matchDis2 = maxLoc3.x - maxLoc2.x;
	int matchDis3 = maxLoc3.x - maxLoc1.x;

	int differ1 = abs(dis1 - matchDis1);
	int differ2 = abs(dis2 - matchDis2);
	int differ3 = abs(dis3 - matchDis3);

	cv::Point maxLoc;			//最佳匹配位置
	int col = 0;
	if (differ1 <= 2)
	{
		maxLoc = maxLoc1;
		col = col1;
	}
	else if (differ2 <= 2)
	{
		maxLoc = maxLoc2;
		col = col2;
	}
	else if (differ3 <= 2)
	{
		maxLoc = maxLoc1;
		col = col1;
	}
	else {
		maxLoc = maxLoc3;
		col = col3;
	}
	//定义拼接图像
	int newCol = maxLoc.x + c / 4;
	int newRow = maxLoc.y - 40;			//y轴方向的偏移度
	cv::Mat dst(speImage01.rows + abs(newRow), speImage01.cols + col - newCol, CV_8UC1);
	if (newRow <= 0) {
		cv::Mat roiLeft = dst(Rect(0, 0, speImage02.cols, speImage02.rows));	//公共区域左部分
		speImage02.copyTo(roiLeft);
	}
	else {
		cv::Mat roiLeft = dst(Rect(0, newRow, speImage02.cols, speImage02.rows));	//公共区域左部分
		speImage02.copyTo(roiLeft);
	}


	////在有图上画出模板区域

	//cv::Mat rDebugImg = speImage01.clone();
	//cv::rectangle(rDebugImg, Rect(newCol, 0, imgTemp.cols, imgTemp.rows), Scalar(255, 255, 0), 2, 8);
	////在左图上画出模板区域
	//cv::Mat lDebugImg = speImage02.clone();
	//cv::rectangle(lDebugImg, Rect(col, 0, imgTemp.cols, imgTemp.rows), Scalar(255, 255, 0), 2, 8);

	//拼接公共区域右半部分
	cv::Mat roiMatched = speImage01(Rect(newCol, 0, speImage01.cols - newCol, speImage01.rows - 1));
	if (newRow >= 0) {
		cv::Mat roiRight = dst(Rect(col, 0, roiMatched.cols, roiMatched.rows));
		roiMatched.copyTo(roiRight);
	}
	else {
		cv::Mat roiRight = dst(Rect(col, abs(newRow), roiMatched.cols, roiMatched.rows));
		roiMatched.copyTo(roiRight);
	}


	////利用加权，进行图像融合处理(公共区域宽度为20)

	//cv::resize(imgTemp, imgTemp, cv::Size(20, imgTemp.rows ));
	////cv::Mat leftTempImg = speImage02(cv::Rect(rectCut));//imgTemp;
	//cv::Mat leftTempImg = speImage02(cv::Rect(rectCut.x, rectCut.y, 20, imgTemp.rows));
	//cv::Mat rightTempImg = speImage01(Rect(newCol, maxLoc.y, 20, imgTemp.rows));
	//cv::Mat mergeImg(imgTemp.size(), imgTemp.type(), cv::Scalar(0));		//融合图像
	//for (int y = 0; y < imgTemp.rows; y++)
	//{
	//	for (int x = 0; x < 20; x++)
	//	{
	//		int leftPixel = leftTempImg.at<uchar>(y, x);
	//		int rightPixel = rightTempImg.at<uchar>(y, x);
	//		mergeImg.at<uchar>(y, x) = (1 - (0.05*x))*leftPixel
	//			+ (0.05*x)*rightPixel;

	//	}

	//}
	////cv::addWeighted(leftTempImg, 0.5, rightTempImg, 0.5, 0, mergeImg);
	//if (newRow <= 0) {
	//	cv::Mat roi = dst(Rect(rectCut.x, 20, imgTemp.cols, imgTemp.rows));
	//	mergeImg.copyTo(roi);
	//}
	//else {
	//	cv::Mat roi = dst(Rect(rectCut.x, 20+ newRow, imgTemp.cols, imgTemp.rows));
	//	mergeImg.copyTo(roi);
	//}



	dstImg.create(dst.size(), dst.type());
	dstImg = dst.clone();
	//	cv::imshow("融合之后", dstImg);

}
//获取模板匹配相关度
void getCorr(cv::Mat &img1, cv::Mat &img2, cv::Mat &matchResult, int angle)
{
	int c = img2.cols;	//为单张展开图像的大约宽度
	int col = (c / 2 + (c*double(angle / 180.0) / 2));		//左图匹配处
	int rCol = c - col;

	//cv::Rect rectCut(col, 0, 20, img2.rows);
	cv::Rect rectCut(col, 40, 20, img2.rows - 45);			//定义模板位置大小
	cv::Rect rectMatched(rCol - 3, 0, 7 + 20, img1.rows);		//定义匹配位置
	//cv::rectangle(img1, rectMatched, Scalar(255, 255, 0), 2, 8);		//在图像上画出位置
	//cv::rectangle(img2, rectCut, Scalar(255, 255, 0), 2, 8);
	////cv::Rect rectMatched((c*11) / 36, 0, (c / 9) + 20, img1.rows);		//定义匹配位置
	cv::Mat imgTemp = img2(rectCut);			//在左图像上取模板
	cv::Mat imgMatched = img1(rectMatched);			//取右图像左半区域进行模板匹配

	int width = imgMatched.cols - imgTemp.cols + 1;
	int height = imgMatched.rows - imgTemp.rows + 1;
	matchResult.create(height, width, CV_32FC1);
	cv::matchTemplate(imgMatched, imgTemp, matchResult, cv::TM_CCORR_NORMED);			//匹配
	//float pixel = matchResult.at<float>(0, 70);
}

//得到最佳匹配点后进行拼接
void StitImg(cv::Mat &img1, cv::Mat &img2, cv::Mat &dstImg, int angle, int BestX, int BestY)
{
	int c = img2.cols;	//为单张展开图像的大约宽度
	int col = (c / 2 + (c*double(angle / 180.0) / 2));
	//定义拼接图像
	int newRow = BestY - 40;
	//int newRow = 0;
	//cv::Mat dst(img1.rows , img1.cols + col - BestX, CV_8UC1);
	//img2(cv::Rect(0, 0, col, img2.rows)).copyTo(dst(cv::Rect(0, 0, col, img2.rows)));
	//img1(cv::Rect(BestX, 0, img1.cols - BestX, img1.rows)).copyTo(dst(cv::Rect(col, 0, img1.cols - BestX, img1.rows)));
	cv::Mat dst(img1.rows + abs(newRow), img1.cols + col - BestX, CV_8UC1);

	if (newRow <= 0) {
		//拼接左图像
		img2(cv::Rect(0, 0, col, img2.rows)).copyTo(dst(cv::Rect(0, 0, col, img2.rows)));
		//拼接公共区域右半部分
		img1(cv::Rect(BestX, 0, img1.cols - BestX, img1.rows)).copyTo
		(dst(cv::Rect(col, abs(newRow), img1.cols - BestX, img1.rows)));
	}
	else {
		//拼接左图像
		img2(cv::Rect(0, 0, col, img2.rows)).copyTo(dst(cv::Rect(0, newRow, col, img2.rows)));
		//拼接公共区域右半部分
		img1(cv::Rect(BestX, 0, img1.cols - BestX, img1.rows)).copyTo(dst(cv::Rect(col, 0, img1.cols - BestX, img1.rows)));
	}


	//进行拼接后的融合
	//cv::Mat leftTempImg = img2(cv::Rect(col, 0, 20, img2.rows ));
	//cv::Mat rightTempImg = img1(Rect(BestX, BestY, 20, img1.rows ));
	cv::Mat leftTempImg = img2(cv::Rect(col, 40, 20, img2.rows - 45));
	cv::Mat rightTempImg = img1(Rect(BestX, BestY, 20, img1.rows - 45));
	cv::Mat mergeImg(leftTempImg.size(), leftTempImg.type(), cv::Scalar(0));		//融合图像
	for (int y = 0; y < leftTempImg.rows; y++)
	{
		for (int x = 0; x < 20; x++)
		{
			int leftPixel = leftTempImg.at<uchar>(y, x);
			int rightPixel = rightTempImg.at<uchar>(y, x);
			mergeImg.at<uchar>(y, x) = (1 - (0.05*x))*leftPixel
				+ (0.05*x)*rightPixel;
		}

	}

	if (newRow <= 0) {
		cv::Mat roi = dst(cv::Rect(col, 40, leftTempImg.cols, leftTempImg.rows));
		mergeImg.copyTo(roi);
	}
	else {
		cv::Mat roi = dst(cv::Rect(col, 40 + newRow, leftTempImg.cols, leftTempImg.rows));
		mergeImg.copyTo(roi);
	}
	//cv::Mat roi = dst(cv::Rect(col, 0, leftTempImg.cols, leftTempImg.rows));
	//mergeImg.copyTo(roi);
	dstImg = dst;
}

//寻找最大互相关
void getMostCorr(std::vector<cv::Mat>&vImg, cv::Mat &allImg, int *arr, int &overallWidth, int &tempWidth)
{
	//建立容器存储模板匹配产生的互相关得分图
	std::vector<cv::Mat>vMatchResult;
	vMatchResult.resize(6);
	for (int i = 0; i < vImg.size() - 1; i++)	//存储前五个得分图
	{
		//获取匹配相关度
		getCorr(vImg[i], vImg[i + 1], vMatchResult[i], arr[i]);
	}
	getCorr(vImg[5], vImg[0], vMatchResult[5], arr[5]);		//存储第六个得分图


	int c = vImg[0].cols;			//图像宽度
	int arrX[6] = { 0 };			//此数组用来存储精确旋转角度在右图中对应的位置
	int sumX = 0;
	for (int i = 0; i < 6; i++)
	{
		arrX[i] = c - (double(arr[i] / 180.0)*c + c) / 2;
		sumX += arrX[i];			//将所有位置进行累加，方便与后面的偏移量x之差和为0
	}

	int arrBestX[6] = { 0 };		//用来存储最佳匹配点的X位置
	int arrBestY[6] = { 0 };		//存储最佳匹配点的Y位置
	float maxScore = 0.0;

	//循环遍历，在六张图像偏移量之后为0的前提下，寻找出匹配得分最大位置
	for (int x1 = arrX[0] - 3; x1 <= arrX[0] + 3; x1++)
	{
		for (int x2 = arrX[1] - 3; x2 <= arrX[1] + 3; x2++)
		{
			for (int x3 = arrX[2] - 3; x3 <= arrX[2] + 3; x3++)
			{
				for (int x4 = arrX[3] - 3; x4 <= arrX[3] + 3; x4++)
				{
					for (int x5 = arrX[4] - 3; x5 <= arrX[4] + 3; x5++)
					{
						for (int x6 = arrX[5] - 3; x6 <= arrX[5] + 3; x6++)
						{
							int arrPixelX[6] = { x1,x2,x3,x4,x5,x6 };		//将六个位置，放入数组，方便后续操作
							int arrPixelY[6] = { 0 };
							if ((x1 + x2 + x3 + x4 + x5 + x6) - sumX == 0)
							{
								float Score = 0.0;
								for (int i = 0; i < 6; i++)
								{

									//cv::Mat matchImg = vMatchResult[i](cv::Rect(arrPixelX[i] -(arrX[i] - 3),0,
									//	1, vMatchResult[i].rows));
									//double minValue, maxValue;
									//cv::Point minLoc, maxLoc;
									//cv::minMaxLoc(matchImg, &minValue, &maxValue, &minLoc, &maxLoc);		//找到最高分的位置
									//vPt.push_back(maxLoc);

									//float maxValue = vMatchResult[i].at<float>(0, (arrPixelX[i] - (arrX[i] - 3) - 1));
									float maxValue = 0.0;
									int maxY = 0;																		//float pixel = 0.0;
									for (int y = 0; y < vMatchResult[0].rows; y++)		//寻找Y方向的最大得分位置
									{

										if (vMatchResult[i].at<float>(y, arrPixelX[i]
											- (arrX[i] - 3)) >= maxValue)
										{
											maxValue = vMatchResult[i].at<float>(y, arrPixelX[i] -
												(arrX[i] - 3));
											maxY = y;
										}
										arrPixelY[i] = maxY;
										//Score+= maxValue;
										Score += maxValue;		//最大得分
									}
								}

								if (Score >= maxScore)		//寻找出最最大得分
								{
									maxScore = Score;
									for (int i = 0; i < 6; i++)
									{
										arrBestX[i] = arrPixelX[i];		//保留下位置信息
										//arrBestY[i] = vPt[i].y;
										arrBestY[i] = arrPixelY[i];

									}
								}


							}
						}
					}
				}
			}
		}
	}

	//进行arrBestY的更新
	for (int i = 0; i < 5; i++)
	{
		if (arrBestY[i] > 40)
		{
			arrBestY[i + 1] = arrBestY[i + 1] + (arrBestY[i] - 40);
		}

		if (arrBestY[i + 1] > 45)
		{
			arrBestY[i + 1] = 45;
		}
	}

	//释放容器
	std::vector<cv::Mat>().swap(vMatchResult);
	//进行拼接
	cv::Mat dst;
	cv::Mat temp;
	StitImg(vImg[0], vImg[1], temp, arr[0], arrBestX[0], arrBestY[0]);
	for (int i = 1; i < 5; i++)
	{
		StitImg(temp, vImg[i + 1], dst, arr[i], arrBestX[i], arrBestY[i]);
		temp = dst;
	}
	//再将第一张与最后一张进行拼接，保证拼接总图像有足够的留余，方便后续处理
	StitImg(temp, vImg[0], dst, arr[5], arrBestX[5], arrBestY[5]);
	//对最终图像进行首尾切割
	////为单张展开图像的大约宽度
	//int col = (c / 2 + (c*double(arr[5] / 180.0) / 2));
	allImg = dst(cv::Rect(c / 4, 0, dst.cols - c / 2, dst.rows));

	overallWidth = allImg.cols - c / 2;		//一个完整周期拼接后的宽度
	tempWidth = c / 4;						//选取模板的宽度，为后续周期调整做准备


}

void findFeaturePoints(cv::Mat &img, int cutX, int width, cv::Point &pt1, cv::Point &pt2)
{
	//转为单通道图像
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	//首先切割图像，去除吸胚部分的影响
	cv::Rect rect(cutX, 0, width, img.rows);
	cv::Mat cutImg = gray(rect);

	//对切割的图像进行二值化处理
	cv::Mat threImg;
	cv::threshold(cutImg, threImg, 200, 255, cv::THRESH_BINARY_INV);

	//通过连通域分析，将其框起来，即确定其上下边界
	//定义轮廓数值
	std::vector<std::vector<cv::Point>>contours;
	cv::findContours(threImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	//删除无效轮廓,找出最大轮廓
	int max = contours[0].size();
	int k = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() >= max) {
			max = contours[i].size();
			k = i;
		}
	}
	//外接矩形
	cv::Rect r0 = cv::boundingRect(contours[k]);

	//寻找特征点
	int topPointY = r0.tl().y;			//最上面点的Y
	int bottomPointY = r0.br().y;		//最下面点的Y

	for (int x = width - 1; x >= 0; x--)
	{
		if (threImg.at<uchar>(topPointY + 5, x) != 0)
		{
			pt1.x = x + cutX;
			pt1.y = topPointY + 5;
			break;
		}
	}
	for (int x = width - 1; x >= 0; x--)
	{
		if (threImg.at<uchar>(bottomPointY - 5, x) != 0)
		{
			pt2.x = x + cutX;
			pt2.y = bottomPointY - 5;
			break;
		}
	}

}

//void ThreadOfCut()
//{
//	std::thread t1(getWhorl, rectImg1, arrCutX[0], width);
//	std::thread t2(getWhorl, rectImg2, arrCutX[1], width);
//	std::thread t3(getWhorl, rectImg3, arrCutX[2], width);
//}

//利用线程进行加速尝试，因为其基础时间耗费不长，线程之间的来回切换反而会加剧耗费时间。无法达到减少时间的作用