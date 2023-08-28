//#include"Stitcher.h"
//
//
////在文件中读取图像
//char * GetFileName(const char * dir, int i, int j)
//{
//
//	char *name = new char[100];
//	if (i < 10) {
//		sprintf(name, "%s\\000%d-00%d.jpg", dir, i, j);
//		return name;
//	}
//	else if (i >= 10) {
//		sprintf(name, "%s\\00%d-00%d.jpg", dir, i, j);
//		return name;
//	}
//
//
//}
//
////螺纹拼接函数
//cv::Mat DoStitcher::Image_Stitching(std::vector<cv::Mat>vImg,int *cutX, int width, int *arrAngle) {
//	std::vector<cv::Mat>adjImg,rectImg,speardImg;
//	
//	image_rectification(vImg, adjImg,width);
//	double start1= (double)getTickCount();
//	getWhorl(adjImg, rectImg, cutX, width);
//	double time1 = ((double)getTickCount()-start1)*1000 / getTickFrequency();
//	cout << "time1:" << time1 << endl;
//	int radius = 0;
//	double start2 = (double)getTickCount();
//	speardCylinder(rectImg, speardImg, radius);
//	double time2 = ((double)getTickCount() - start2) * 1000 / getTickFrequency();
//	cout << "time2:" << time2 << endl;
//	cv::Mat resultImg;
//	int overallWidth = 0;					//用来存放一个完整周期的宽度
//	int tempWidth = 0;
//	double start3 = (double)getTickCount();
//	getMostCorr(speardImg, resultImg, arrAngle, overallWidth, tempWidth);
//	double time3 = ((double)getTickCount() - start3) * 1000 / getTickFrequency();
//	cout << "time3:" << time3 << endl;
//	return resultImg;
//}
//
////1.图像修正函数（采集的图像螺纹部分可能会出现歪斜情况）
//void DoStitcher::image_rectification(std::vector<cv::Mat>vImg, std::vector<cv::Mat>&adjImg , int width)
//{
//	//std::vector<cv::Mat>adjImg;
//	adjImg.resize(vImg.size());
//#pragma omp critical
//	for (int i = 0; i < vImg.size(); i++)
//	{
//		cv::Point p1 = cv::Point(0, 0);
//		cv::Point p2 = cv::Point(0, 0);
//		findFeaturePoints(vImg[i], cutAdjX, width, p1, p2);
//		adjustImg(vImg[i], adjImg[i], p1, p2);
//	}
//}
//
////1.1 寻找矫正需要的特征点
//void DoStitcher::findFeaturePoints(cv::Mat &img, int cutX, int width, cv::Point &pt1, cv::Point &pt2)
//{
//	//首先切割图像，去除吸胚部分的影响
//	cv::Rect rect(cutX, 0, adjWidth, img.rows);
//	cv::Mat cutImg = img(rect);
//
//	//对切割的图像进行二值化处理
//	cv::Mat threImg;
//	cv::threshold(cutImg, threImg, 200, 255, cv::THRESH_BINARY_INV);
//
//	//通过连通域分析，将其框起来，即确定其上下边界
//	//定义轮廓数值
//	std::vector<std::vector<cv::Point>>contours;
//	cv::findContours(threImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//
//	//删除无效轮廓,找出最大轮廓
//	int max = contours[0].size();
//	int k = 0;
//	for (int i = 0; i < contours.size(); i++)
//	{
//		if (contours[i].size() >= max) {
//			max = contours[i].size();
//			k = i;
//		}
//	}
//	//外接矩形
//	cv::Rect r0 = cv::boundingRect(contours[k]);
//
//	//寻找特征点
//	int topPointY = r0.tl().y;			//最上面点的Y
//	int bottomPointY = r0.br().y;		//最下面点的Y
//
//	for (int x = adjWidth - 1; x >= 0; x--)
//	{
//		if (threImg.at<uchar>(topPointY + 5, x) != 0)
//		{
//			pt1.x = x + cutX;
//			pt1.y = topPointY + 5;
//			break;
//		}
//	}
//	for (int x = adjWidth - 1; x >= 0; x--)
//	{
//		if (threImg.at<uchar>(bottomPointY - 5, x) != 0)
//		{
//			pt2.x = x + cutX;
//			pt2.y = bottomPointY - 5;
//			break;
//		}
//	}
//}
//
////1.2 对图像进行角度修正
//void DoStitcher::adjustImg(cv::Mat &img, cv::Mat &adjImg, cv::Point pt1, cv::Point pt2)
//{
//	//计算两点x方向的偏移，进行平移校正
//	img.copyTo(adjImg);
//	cv::Point center(((pt1.x + pt2.x) / 2), ((pt1.y + pt2.y) / 2));		//两点连线的中心位置
//	double length = (pt2.y - pt1.y) / 2;
//
//	double num = pt1.x - pt2.x;			//差的像素数量
//
//	double moveNum = num / double(pt2.y - pt1.y);
//	//处理中心点的上半部分
//	for (int y = pt1.y; y < center.y; ++y)
//	{
//		double move = double(center.y - y)*moveNum;
//		int move1 = int(move);
//		double weight = abs(move - move1);
//		for (int x = 300; x < 600; ++x)
//		{
//
//			if (num > 0) {
//				int value1 = (1.0 - weight) * (img.at<uchar>(y, x + move1));
//				int value2 = (weight)*(img.at<uchar>(y, x + move1 + 1));
//				adjImg.at<uchar>(y, x) = value1 + value2;
//			}
//			else {
//				int value1 = (1.0 - weight) * (img.at<uchar>(y, x + move1));
//				int value2 = (weight)*(img.at<uchar>(y, x + move1 - 1));
//				adjImg.at<uchar>(y, x) = value1 + value2;
//			}
//		}
//	}
//
//	//处理中心点的下半部分
//	for (int y = center.y; y < pt2.y; ++y)
//	{
//		double move = double(y - center.y)*moveNum;
//		int move1 = int(move);
//		double weight = abs(move - move1);
//		for (int x = 300; x < 600; ++x)
//		{
//			if (num > 0) {
//				int value1 = (1.0 - weight) * (img.at<uchar>(y, x - move1));
//				int value2 = (weight)*(img.at<uchar>(y, x - move1 - 1));
//				adjImg.at<uchar>(y, x) = value1 + value2;
//			}
//			else {
//				int value1 = (1.0 - weight) * (img.at<uchar>(y, x - move1));
//				int value2 = (weight)*(img.at<uchar>(y, x - move1 + 1));
//				adjImg.at<uchar>(y, x) = value1 + value2;
//			}
//		}
//	}
//}
//
////2.图像切割函数（将螺纹区域切割处理）
////通过自定义框，选中螺纹区域，之后将子区域提取出来
//void DoStitcher::getWhorl(std::vector<cv::Mat>adjImg, std::vector<cv::Mat>&rectImg, int *arrCutX, int width)
//{
//	rectImg.resize(adjImg.size());
//#pragma omp parallel for
//	for (int i = 0; i < adjImg.size(); i++)
//	{
//		cv::Rect rect(arrCutX[i/2], 0, width, adjImg[i].rows);
//		cv::Mat rectImage01 = adjImg[i](rect);
//		//cv::Rect rect(360, 0, 168, image.rows);
//		//rectImage01 = image(rect);
//
//		//对已经切割的图像进行二值化处理
//		cv::Mat threImage;
//		cv::threshold(rectImage01, threImage, 200, 255, cv::THRESH_BINARY_INV);
//
//		////利用形态学方法对边角进行修整
//		//cv::Mat erodedImage;
//		//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
//		//cv::erode(threImage, erodedImage, element, cv::Point(-1, -1), 5);		//腐蚀
//
//		////对其进行膨胀处理，将间隙填充
//		//cv::Mat dilatedImage;
//		//cv::dilate(erodedImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 1);
//		//通过连通域分析，将其框起来，即确定其上下边界
//		//定义轮廓数值
//		std::vector<std::vector<cv::Point>>contours;
//		cv::findContours(threImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//
//		//删除无效轮廓,找出最大轮廓
//		int max = contours[0].size();
//		int k = 0;
//		for (int i = 0; i < contours.size(); i++)
//		{
//			if (contours[i].size() >= max) {
//				max = contours[i].size();
//				k = i;
//			}
//		}
//		//切割
//		cv::Rect r0 = cv::boundingRect(contours[k]);
//		int centerY = r0.height / 2 + r0.y;
//		int leftBegin = centerY - 178;
//		//int leftBegin = r0.tl().y;
//		//cv::Rect r1(r0.x - 100 + x, r0.y, r0.width + 300, r0.height);
//		cv::Rect r1(r0.x - 55 + arrCutX[i/2], leftBegin, r0.width + 55, 356);
//		//cv::Rect r1(r0.x - 55 + x, leftBegin, r0.width + 55, r0.height);
//		//rectImage = rectImage01(r0);
//		cv::Mat rectImage02 = adjImg[i](r1);
//		//进行旋转调整
//		cv::Mat temp;
//		transpose(rectImage02, temp);
//		flip(temp, rectImg[i], 1);
//		//cv::flip(rectImage02, rectImg[i], 1);
//		//rotate(rectImage02, rectImg[i], 270);
//	}
//}
//
////2.2 //将输入的图片逆时针旋转
//void DoStitcher::rotate(cv::Mat &image, cv::Mat &rectImage, int angle)
//{
//	Point center(image.cols / 2, image.rows / 2); //旋转中心
//	Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
//	warpAffine(image, rectImage, rotMat, Size(image.rows, image.cols));
//	
//	//float radian = (float)(0.5 * CV_PI);
//
//	////填充图像
//	//int maxBorder = (int)(max(image.cols, image.rows)* 1.414); //即为sqrt(2)*max
//	//int dx = (maxBorder - image.cols) / 2;
//	//int dy = (maxBorder - image.rows) / 2;
//	//copyMakeBorder(image, rectImage, dy, dy, dx, dx, BORDER_CONSTANT);
//	////旋转
//	//Point2f center((float)(rectImage.cols / 2), (float)(rectImage.rows / 2));
//	//Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);//求得旋转矩阵
//	//warpAffine(rectImage, rectImage, affine_matrix, rectImage.size());
//	////计算图像旋转之后包含图像的最大的矩形
//	//float sinVal = abs(sin(radian));
//	//float cosVal = abs(cos(radian));
//	//Size targetSize((int)(image.cols * cosVal + image.rows * sinVal),
//	//	(int)(image.cols * sinVal + image.rows * cosVal));
//
//	////剪掉多余边框
//	//int x = (rectImage.cols - targetSize.width) / 2;
//	//int y = (rectImage.rows - targetSize.height) / 2;
//	//Rect rect(x, y, targetSize.width, targetSize.height);
//	//rectImage = Mat(rectImage, rect);
//
//}
//
////3.对螺纹部分进行柱体展开
//void DoStitcher::speardCylinder(std::vector<cv::Mat>rectImg, std::vector<cv::Mat>&speardImg, int radius)
//{
//	//3.1 获取展开矩阵
//	cv::Mat mapX, mapY;
//	speardCylinderMatrix(rectImg[0], radius, mapX, mapY);
//	speardImg.resize(rectImg.size());
//#pragma omp parallel for
//	for (int i = 0; i < rectImg.size(); i++)
//	{
//		//利用重投影矩阵，进行柱体展开
//		cv::remap(rectImg[i], speardImg[i], mapX, mapY, cv::INTER_LINEAR);
//	}
//}
//
////3.1获取展开矩阵
//void DoStitcher::speardCylinderMatrix(cv::Mat &rectImage01, int &radius, cv::Mat &mapX, cv::Mat &mapY)
//{
//
//	radius = rectImage01.cols / 2.0;		//定义半径，为子图像宽的一半
//	cv::Mat speImage(rectImage01.rows, PI*radius, CV_8UC1);
//
//	//映射参数
//	mapX.create(speImage.size(), CV_32F);
//	mapY.create(speImage.size(), CV_32F);
//
//	//创建映射参数
//	for (int y = 0; y < speImage.rows; y++)
//	{
//		for (int x = 0; x < speImage.cols; x++)
//		{
//
//			mapY.at<float>(y, x) = (y);
//			mapX.at<float>(y, x) = (radius*sin(((x - speImage.cols / 2.0)) / radius) + radius);
//
//		}
//	}
//	//cv::remap(rectImage, speImage, mapX, mapY, cv::INTER_LINEAR);
//	//	cv::imshow("重投影", speImage);
//}
//
////4.利用模板匹配，对展开图像进行图像拼接
//void DoStitcher::getMostCorr(std::vector<cv::Mat>&vImg, cv::Mat &allImg, int *arr, int &overallWidth, int &tempWidth)
//{
//	//建立容器存储模板匹配产生的互相关得分图
//	std::vector<cv::Mat>vMatchResult;
//	vMatchResult.resize(6);
////#pragma omp parallel for
//	for (int i = 0; i < vImg.size() - 1; i++)	//存储前五个得分图
//	{
//		//获取匹配相关度
//		getCorr(vImg[i], vImg[i + 1], vMatchResult[i], arr[i]);
//	}
//
//	getCorr(vImg[5], vImg[0], vMatchResult[5], arr[5]);		//存储第六个得分图
//
//
//	int c = vImg[0].cols;			//图像宽度
//	int arrX[6] = { 0 };			//此数组用来存储精确旋转角度在右图中对应的位置
//	int sumX = 0;
////#pragma omp parallel for
//	for (int i = 0; i < 6; i++)
//	{
//		arrX[i] = c - (double(arr[i] / 180.0)*c + c) / 2;
//		sumX += arrX[i];
//	}
//	//sumX = arrX[0]+ arrX[1]+ arrX[2]+ arrX[3]+ arrX[4]+ arrX[5];			//将所有位置进行累加，方便与后面的偏移量x之差和为0
//	int arrBestX[6] = { 0 };		//用来存储最佳匹配点的X位置
//	int arrBestY[6] = { 0 };		//存储最佳匹配点的Y位置
//	float maxScore = 0.0;
//
//	//循环遍历，在六张图像偏移量之后为0的前提下，寻找出匹配得分最大位置
//#pragma omp parallel for
//	for (int x1 = arrX[0] - 3; x1 <= arrX[0] + 3; x1++)
//	{
//		for (int x2 = arrX[1] - 3; x2 <= arrX[1] + 3; x2++)
//		{
//			for (int x3 = arrX[2] - 3; x3 <= arrX[2] + 3; x3++)
//			{
//				for (int x4 = arrX[3] - 3; x4 <= arrX[3] + 3; x4++)
//				{
//					for (int x5 = arrX[4] - 3; x5 <= arrX[4] + 3; x5++)
//					{
//						for (int x6 = arrX[5] - 3; x6 <= arrX[5] + 3; x6++)
//						{
//							int arrPixelX[6] = { x1,x2,x3,x4,x5,x6 };		//将六个位置，放入数组，方便后续操作
//							int arrPixelY[6] = { 0 };
//							if ((x1 + x2 + x3 + x4 + x5 + x6) - sumX == 0)
//							{
//								float Score = 0.0;
//								for (int i = 0; i < 6; i++)
//								{
//
//									//cv::Mat matchImg = vMatchResult[i](cv::Rect(arrPixelX[i] -(arrX[i] - 3),0,
//									//	1, vMatchResult[i].rows));
//									//double minValue, maxValue;
//									//cv::Point minLoc, maxLoc;
//									//cv::minMaxLoc(matchImg, &minValue, &maxValue, &minLoc, &maxLoc);		//找到最高分的位置
//									//vPt.push_back(maxLoc);
//
//									//float maxValue = vMatchResult[i].at<float>(0, (arrPixelX[i] - (arrX[i] - 3) - 1));
//									float maxValue = 0.0;
//									int maxY = 0;																		//float pixel = 0.0;
//									for (int y = 0; y < vMatchResult[0].rows; y++)		//寻找Y方向的最大得分位置
//									{
//
//										if (vMatchResult[i].at<float>(y, arrPixelX[i]
//											- (arrX[i] - 3)) >= maxValue)
//										{
//											maxValue = vMatchResult[i].at<float>(y, arrPixelX[i] -
//												(arrX[i] - 3));
//											maxY = y;
//										}
//										arrPixelY[i] = maxY;
//										//Score+= maxValue;
//										Score += maxValue;		//最大得分
//									}
//								}
//
//								if (Score >= maxScore)		//寻找出最最大得分
//								{
//									maxScore = Score;
//									for (int i = 0; i < 6; i++)
//									{
//										arrBestX[i] = arrPixelX[i];		//保留下位置信息
//										//arrBestY[i] = vPt[i].y;
//										arrBestY[i] = arrPixelY[i];
//
//									}
//								}
//
//
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//
//	//进行arrBestY的更新
////#pragma omp parallel for
//	for (int i = 0; i < 5; i++)
//	{
//		if (arrBestY[i] > 40)
//		{
//			arrBestY[i + 1] = arrBestY[i + 1] + (arrBestY[i] - 40);
//		}
//
//		if (arrBestY[i + 1] > 45)
//		{
//			arrBestY[i + 1] = 45;
//		}
//	}
//
//	//释放容器
//	std::vector<cv::Mat>().swap(vMatchResult);
//	//进行拼接
//	cv::Mat dst;
//	cv::Mat temp;
//	StitImg(vImg[0], vImg[1], temp, arr[0], arrBestX[0], arrBestY[0]);
////#pragma omp parallel for
//	for (int i = 1; i < 5; i++)
//	{
//		StitImg(temp, vImg[i + 1], dst, arr[i], arrBestX[i], arrBestY[i]);
//		temp = dst;
//	}
//	//再将第一张与最后一张进行拼接，保证拼接总图像有足够的留余，方便后续处理
//	StitImg(temp, vImg[0], dst, arr[5], arrBestX[5], arrBestY[5]);
//	//对最终图像进行首尾切割
//	////为单张展开图像的大约宽度
//	//int col = (c / 2 + (c*double(arr[5] / 180.0) / 2));
//	allImg = dst(cv::Rect(c / 4, 0, dst.cols - c / 2, dst.rows));
//
//	overallWidth = allImg.cols - c / 2;		//一个完整周期拼接后的宽度
//	tempWidth = c / 4;						//选取模板的宽度，为后续周期调整做准备
//
//
//}
//
////4.1 //获取模板匹配相关度
//void DoStitcher::getCorr(cv::Mat &img1, cv::Mat &img2, cv::Mat &matchResult, int angle)
//{
//	int c = img2.cols;	//为单张展开图像的大约宽度
//	int col = (c / 2 + (c*double(angle / 180.0) / 2));		//左图匹配处
//	int rCol = c - col;
//
//	//cv::Rect rectCut(col, 0, 20, img2.rows);
//	cv::Rect rectCut(col, 40, 20, img2.rows - 45);			//定义模板位置大小
//	cv::Rect rectMatched(rCol - 3, 0, 7 + 20, img1.rows);		//定义匹配位置
//	//cv::rectangle(img1, rectMatched, Scalar(255, 255, 0), 2, 8);		//在图像上画出位置
//	//cv::rectangle(img2, rectCut, Scalar(255, 255, 0), 2, 8);
//	////cv::Rect rectMatched((c*11) / 36, 0, (c / 9) + 20, img1.rows);		//定义匹配位置
//	cv::Mat imgTemp = img2(rectCut);			//在左图像上取模板
//	cv::Mat imgMatched = img1(rectMatched);			//取右图像左半区域进行模板匹配
//
//	int width = imgMatched.cols - imgTemp.cols + 1;
//	int height = imgMatched.rows - imgTemp.rows + 1;
//	matchResult.create(height, width, CV_32FC1);
//	cv::matchTemplate(imgMatched, imgTemp, matchResult, cv::TM_CCORR_NORMED);			//匹配
//	//float pixel = matchResult.at<float>(0, 70);
//}
//
////4.2 //得到最佳匹配点后进行拼接
//void DoStitcher::StitImg(cv::Mat &img1, cv::Mat &img2, cv::Mat &dstImg, int angle, int BestX, int BestY)
//{
//	int c = img2.cols;	//为单张展开图像的大约宽度
//	int col = (c / 2 + (c*double(angle / 180.0) / 2));
//	//定义拼接图像
//	int newRow = BestY - 40;
//	//int newRow = 0;
//	//cv::Mat dst(img1.rows , img1.cols + col - BestX, CV_8UC1);
//	//img2(cv::Rect(0, 0, col, img2.rows)).copyTo(dst(cv::Rect(0, 0, col, img2.rows)));
//	//img1(cv::Rect(BestX, 0, img1.cols - BestX, img1.rows)).copyTo(dst(cv::Rect(col, 0, img1.cols - BestX, img1.rows)));
//	cv::Mat dst(img1.rows + abs(newRow), img1.cols + col - BestX, CV_8UC1);
//
//	if (newRow <= 0) {
//		//拼接左图像
//		img2(cv::Rect(0, 0, col, img2.rows)).copyTo(dst(cv::Rect(0, 0, col, img2.rows)));
//		//拼接公共区域右半部分
//		img1(cv::Rect(BestX, 0, img1.cols - BestX, img1.rows)).copyTo
//		(dst(cv::Rect(col, abs(newRow), img1.cols - BestX, img1.rows)));
//	}
//	else {
//		//拼接左图像
//		img2(cv::Rect(0, 0, col, img2.rows)).copyTo(dst(cv::Rect(0, newRow, col, img2.rows)));
//		//拼接公共区域右半部分
//		img1(cv::Rect(BestX, 0, img1.cols - BestX, img1.rows)).copyTo(dst(cv::Rect(col, 0, img1.cols - BestX, img1.rows)));
//	}
//
//
//	//进行拼接后的融合
//	//cv::Mat leftTempImg = img2(cv::Rect(col, 0, 20, img2.rows ));
//	//cv::Mat rightTempImg = img1(Rect(BestX, BestY, 20, img1.rows ));
//	cv::Mat leftTempImg = img2(cv::Rect(col, 40, 20, img2.rows - 45));
//	cv::Mat rightTempImg = img1(Rect(BestX, BestY, 20, img1.rows - 45));
//	cv::Mat mergeImg(leftTempImg.size(), leftTempImg.type(), cv::Scalar(0));		//融合图像
//#pragma omp critical
//	for (int y = 0; y < leftTempImg.rows; y++)
//	{
//		for (int x = 0; x < 20; x++)
//		{
//			int leftPixel = leftTempImg.at<uchar>(y, x);
//			int rightPixel = rightTempImg.at<uchar>(y, x);
//			mergeImg.at<uchar>(y, x) = (1 - (0.05*x))*leftPixel
//				+ (0.05*x)*rightPixel;
//		}
//
//	}
//
//	if (newRow <= 0) {
//		cv::Mat roi = dst(cv::Rect(col, 40, leftTempImg.cols, leftTempImg.rows));
//		mergeImg.copyTo(roi);
//	}
//	else {
//		cv::Mat roi = dst(cv::Rect(col, 40 + newRow, leftTempImg.cols, leftTempImg.rows));
//		mergeImg.copyTo(roi);
//	}
//	//cv::Mat roi = dst(cv::Rect(col, 0, leftTempImg.cols, leftTempImg.rows));
//	//mergeImg.copyTo(roi);
//	dstImg = dst;
//}
//
////做一个首尾切割函数，确定周期，对拼接图像进行首尾切割，使其在一个周期
//void DoStitcher::getPeriod(cv::Mat &img01, cv::Mat &img02, int angle, int &dis, cv::Point &p)
//{
//	//首先根据旋转角度，确定近似匹配位置
//	//在匹配左图的匹配位置列为
//	int c = img01.cols;	//为单张展开图像的大约宽度
//	//利用旋转角度，确定大概的拼接位置
//
//	int col = (c / 2 + (c*double(angle / 180.0) / 2));
//	//int col= (img01.cols / 2 + (img01.cols *double(angle / 180.0) / 2));
//	dis = img01.cols - col;			//此列到尾列的距离
//
//	//定义模板尺寸以及匹配位置
//	cv::Mat tempImg = img01(Rect(col, 20, 50, img01.rows - 25));
//	cv::Mat matchImg = img02(Rect(c / 4, 0, c / 3, img02.rows));
//
//	//模板匹配
//	cv::Mat result(matchImg.rows - tempImg.rows + 1, matchImg.cols - tempImg.cols + 1, CV_32FC1);
//	cv::matchTemplate(matchImg, tempImg, result, cv::TM_CCORR_NORMED);
//	cv::normalize(result, result, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围
//
//	//找到最相似的位置
//	double minVal, maxVal;
//	cv::Point minPt, maxPt;
//	cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);
//	p.x = maxPt.x + c / 4;
//
//	////在有图上画出模板区域
//
//	//cv::Mat rDebugImg = img02.clone();
//	//cv::rectangle(rDebugImg, Rect(maxPt.x, maxPt.y, tempImg.cols, tempImg.rows), Scalar(255, 255, 0), 2, 8);
//	////在左图上画出模板区域
//	//cv::Mat lDebugImg = img01.clone();
//	//cv::rectangle(lDebugImg, Rect(col, 0, tempImg.cols, tempImg.rows), Scalar(255, 255, 0), 2, 8);
//
//
//}