//#include"myStitcher.h"
//
//
////相似度匹配算法之灰度值方差匹配法：
//double get_variance(Mat&a, Mat&b)
//{
//	if (a.rows != b.rows || a.cols != b.cols || a.channels() != b.channels())
//	{
//		printf("not the same size!\n");
//		return 0;
//	}
//
//	//处理图像相似度
//	//1.求出每一行到灰度值均值,加入容器，作为特征值；
//	//2.求出灰度值总平均值与每行平均值的方差；
//	//3.行行比较与模版方差的接近程度
//
//
//	vector<double> variance_a;
//	vector<double> variance_b;
//	double var_a = 0;
//	double var_b = 0;
//	double sum_a = 0;
//	double sum_b = 0;
//	double mean_a;
//	double mean_b;
//	double sum_variance = 0.0;
//	//将每行灰度值均值存入容器
//	for (int i = 0; i < a.rows; i++) {
//		mean_a = 0;
//		mean_b = 0;
//		for (int j = 0; j < a.cols; j++) {
//			mean_a += a.at<uchar>(i, j);
//			mean_b += b.at<uchar>(i, j);
//		}
//		mean_a /= (double)(a.rows*a.cols);
//		mean_b /= (double)(a.rows*a.cols);
//		sum_a += mean_a;
//		sum_b += mean_b;
//		variance_a.push_back(mean_a);
//		variance_b.push_back(mean_b);
//	}
//	//全图灰度值均值
//	mean_a = sum_a / (double)variance_a.size();
//	mean_b = sum_b / (double)variance_b.size();
//	//灰度值方差之差累加
//	for (int i = 0; i < variance_a.size(); i++) {
//		var_a = (variance_a[i] - mean_a)*(variance_a[i] - mean_a);
//		var_b = (variance_b[i] - mean_b)*(variance_b[i] - mean_b);
//		sum_variance += abs(var_a - var_b);
//	}
//
//	return sum_variance;
//}
//
//
//
///*自适应模板匹配,切割螺纹子图像*/
//void advTemplateMatch(cv::Mat &image, cv::Mat &my_template)
//{
//	int best_index;	//存储最佳匹配的序号
//	double  min_diff;	//存储最小的方差值之差
//	Rect best_rect;		//存储最佳的匹配框
//	double *minVal, *maxVal;
//	cv::Point minPt, maxPt;
//	Mat temp_template;			//改变 后的模板
//	//循环缩放，j假设当前模版为最小尺寸，每次循环放大5%，循环100次
//	for (int index = 0; index < 100; index++)
//	{
//		//定义模板
//		//获得缩放后的模版
//		temp_template = my_template.clone();
//		int new_rows = my_template.rows + index * 0.05*my_template.rows;
//		int new_cols = my_template.cols + index * 0.05 *my_template.cols;
//		resize(temp_template, temp_template, Size(new_cols, new_rows));
//
//		//进行模板匹配
//		cv::Mat result;
//		result.create(image.dims, image.size, image.type());
//		cv::matchTemplate(image, temp_template, result, cv::TM_SQDIFF);
//
//		//获取模版匹配得到的rect
//		Point minPoint;
//		Point maxPoint;
//		double *minVal = NULL;
//		double *maxVal = NULL;
//		minMaxLoc(result, minVal, maxVal, &minPoint, &maxPoint);
//		Rect rect(minPoint.x, minPoint.y, new_cols, new_rows);
//
//		//获取匹配部分的roi图像
//		Mat result_img = image.clone();
//		Mat result_img_roi = result_img(rect);
//		//相似度比较部分：
//		//灰度值方差的相似度比较
//		//variance_diff表示灰度值方差，方差越小，相似度越高；
//		double variance_diff = get_variance(result_img_roi, temp_template);
//
//		//默认值为index=0时获取的值；方便与之后的值最比较
//		if (index == 0) {
//			min_diff = variance_diff;
//			best_index = index;
//			best_rect = rect;
//		}
//		//当前值与目前的最小方差做比较
//		if (variance_diff < min_diff) {
//			min_diff = variance_diff;
//			best_index = index;
//			best_rect = rect;
//		}
//
//
//
//	}
//	//在相似度最高的位置切割矩形
//
//	cv::Mat rectImage = image(best_rect);
//	//cv::imshow("模板图像", target);
//	cv::imshow("子图像", rectImage);
//}
//
//void getThreadSubimage(cv::Mat &image, cv::Mat &rectImage)
//{
//	int firstY;			//第一个获得的像素y位置
//	int lastY;			//最后一个像素y位置
//
//	cvtColor(image, image, COLOR_BGR2GRAY);//先转为单通道图像
//
//	//图像太大，对其进行放缩处理
//	cv::Mat shrImage;
//	cv::resize(image, shrImage, shrImage.size(), 0.5, 0.5, 1);
//
//	//对图像进行二值化处理
//	cv::Mat threImage;
//	cv::threshold(shrImage, threImage, 127, 255, cv::THRESH_BINARY);
//
//	cv::morphologyEx(threImage, threImage, cv::MORPH_OPEN, cv::Mat());//开运算处理，消除细小噪声点
//	//cv::namedWindow("二值化");
//	cv::imshow("二值化", threImage);
//
//	//遍历二值图像最后一行的像素
//	//从前向后遍历
//	for (int y = 0; y < threImage.cols; y++)
//	{
//		//得到第一个0灰度位置,保存退出
//		if (threImage.at<uchar>(threImage.rows - 1, y) == 0 &&
//			threImage.at<uchar>(threImage.rows - 1, y + 1) == 0)
//		{
//			firstY = y;
//			break;
//		}
//	}
//	//从后向前遍历
//	for (int y = threImage.cols - 1; y >= 0; y--)
//	{
//		//得到第一个0灰度位置(倒数第一个0位置),保存退出
//		if (threImage.at<uchar>(threImage.rows - 1, y) == 0 &&
//			threImage.at<uchar>(threImage.rows - 1, y - 1) == 0)
//		{
//			lastY = y;
//			break;
//		}
//	}
//
//	//切割图像，获取螺旋区域的子图像
//	cv::Rect rect(firstY, 0, lastY - firstY, threImage.rows - 1);
//	rectImage = shrImage(rect);
//	cv::imshow("子图像", rectImage);
//}
//
////获取子图像
//void getRectImage(cv::Mat &img, cv::Mat rectImage01)
//{
//	cv::Mat thrImage;			//存放缩放后的图像
//	cv::Mat image;			//存放滤波之后的图像
//	cv::resize(img, thrImage, thrImage.size(), 0.5, 0.5, 1);
//	int row = thrImage.rows;
//	int col = thrImage.cols;
//	int arr01[2000] = { 0 };			//定义数组存储二值图像每行0像素的数量
//	int arr02[2000] = { 0 };			//定义数组存储数组arr01相邻元素的差值
//	int max;
//	int *idx = NULL;				//指向最小值所在的数组位置
//	int *preIdx = NULL;			//定义指针，指向次小值所在的数组位置
//	int *head = NULL;
//	cv::cvtColor(thrImage, thrImage, COLOR_BGR2GRAY);			//转换为单通道图像
//	cv::GaussianBlur(thrImage, image, cv::Size(5, 5), 3, 3);	//高斯滤波去除噪声影响
//	cv::Mat threImage;
//	cv::threshold(image, threImage, 70, 255, cv::THRESH_BINARY);		//二值化处理
//	cv::erode(threImage, threImage, cv::Mat(), cv::Point(-1, -1), 3);
//	cv::imshow("二值图像", threImage);
//	for (int y = 0; y < row; y++) {			//进行二值图像0像素值每行数量统计
//		int count = 0;
//		for (int x = 0; x < col; x++) {
//			if (threImage.at<uchar>(y, x) == 0) {
//				count++;
//			}
//		}
//		arr01[y] = count;
//	}
//	for (int i = 0; i < row - 1; i++) {			//统计数组相邻的差值
//		arr02[i] = arr01[i] - arr01[i + 1];
//	}
//	max = arr02[0];
//	head = arr02;
//	int y1 = 0;
//	int y2 = 0;			//获取所要求出差值最大或次大所在的行数
//	if (arr01[0] == row || arr01[1] == row) {			//若图像最上方存在黑影，取次小值
//		idx = arr02;
//		for (int j = 0; j < row - 1; j++) {
//			if (arr02[j] >= max) {
//				max = arr02[j];
//				preIdx = idx;
//				idx = arr02 + j;
//
//			}
//		}
//		y1 = preIdx - head;
//		cv::Rect rect(0, y1, col, row - y1 - 1);
//		rectImage01 = thrImage(rect);
//		//	cout << y1 << endl;
//
//	}
//
//	else {			//否则，取最小值
//		for (int j = 0; j < row - 1; j++) {
//			if (arr02[j] >= max) {
//				max = arr02[j];
//				idx = arr02 + j;
//			}
//		}
//		y2 = idx - head;
//		cv::Rect rect(0, y2, col, row - y2 - 1);
//		rectImage01 = thrImage(rect);
//		//cout << y2 << endl;
//	}
//
//
//	cv::imshow("子图像", rectImage01);
//
//}
//
////提取模板图像，进行模板匹配进行，子图像获取
//void getTempImage(cv::Mat &image)
//{
//	cv::Rect rect(175, 342, 346, 173);
//	cv::Mat tempImage = image(rect);
//	cv::imwrite("D:\\files\\c++文件夹\\c++ files\\testStitcher\\testStitcher\\tempImg.jpg", tempImage);
//
//}
//
////利用模板匹配，进行螺纹区域子图像提取
//void myRectImage(cv::Mat &target, cv::Mat &image, cv::Mat &rectImage)
//{
//	cv::Mat result;
//	//进行模板匹配
//	cv::matchTemplate(image, target, result, cv::TM_SQDIFF);
//
//	//找到最相似的位置
//	double minVal, maxVal;
//	cv::Point minPt, maxPt;
//	cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);
//
//	//在相似度最高的地方进行切割
//	cv::Rect rect(minPt.x, minPt.y, target.cols, target.rows);
//	rectImage = image(rect);
//
//	cv::imshow("子图像", rectImage);
//}
//
////利用连通域分析，进行子图像切割
//void connectImage(cv::Mat &img)
//{
//	cv::Mat image;
//	cv::resize(img, image, image.size(), 0.5, 0.5, 1);
//	//转换为灰度图像
//	cv::Mat grayImage;
//	cvtColor(image, grayImage, COLOR_BGR2GRAY);
//
//	//二值化处理
//	cv::Mat threImage;
//	cv::threshold(grayImage, threImage, 70, 255, cv::THRESH_BINARY);
//
//	//使用时间种子，随机生成不同的颜色区分不同的连通域
//	cv::RNG rng((unsigned)time(NULL));
//	cv::Mat outImage;
//	int counts = cv::connectedComponents(threImage, outImage, 8, CV_16U);
//	std::vector<cv::Vec3b>colors;
//	for (int i = 0; i < counts; i++)
//	{
//		cv::Vec3b vec3 = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
//		colors.push_back(vec3);
//	}
//
//	cv::Mat dst = Mat::zeros(grayImage.size(), image.type());
//	int width = dst.cols;
//	int height = dst.rows;
//	for (int row = 0; row < height; row++) {
//		for (int col = 0; col < width; col++) {
//			int label = outImage.at<uint16_t>(row, col);
//			if (label == 0)continue;
//			dst.at<Vec3b>(row, col) = colors[label];
//		}
//	}
//	cv::imshow("processed", dst);
//
//}
//
////通过图像的轮廓，对螺纹区域的子图像进行提取
//void getContours(cv::Mat &image)
//{
//	//定义轮廓数值
//	std::vector<vector<cv::Point>>contours;
//	std::vector<cv::Vec4i>hierarchy;
//
//	//图像太大，首先对图像进行放缩处理
//	cv::Mat thrImage;
//	cv::resize(image, thrImage, thrImage.size(), 0.5, 0.5, 1);
//
//	//对图像进行灰度处理
//	cv::Mat grayImage;
//	cv::cvtColor(thrImage, grayImage, COLOR_BGR2GRAY);
//
//	//对图像进行高斯模糊去噪
//	cv::Mat blurImage;
//	cv::GaussianBlur(grayImage, blurImage, cv::Size(3, 3), 0);
//
//	//对图像进行二值化处理
//	int blockSize = 5;
//	int constValue = 10;
//	cv::Mat threImage;
//	cv::adaptiveThreshold(blurImage, threImage, 255, cv::ADAPTIVE_THRESH_MEAN_C,
//		cv::THRESH_BINARY, blockSize, constValue);
//	cv::imshow("adaptiveThreshold", threImage);
//	//寻找轮廓
//
//	cv::findContours(threImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
//	cv::Mat Contours = Mat::zeros(thrImage.size(), CV_8UC1);
//	cv::Mat contoursImage = Mat::zeros(thrImage.size(), CV_8UC1);
//	for (int i = 0; i < contours.size(); i++)
//	{
//		//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数 
//		for (int j = 0; j < contours[i].size(); j++)
//		{
//			//绘制出contours向量内所有的像素点  
//			cv::Point p = Point(contours[i][j].x, contours[i][j].y);
//			Contours.at<uchar>(p) = 255;
//		}
//		cv::drawContours(contoursImage, contours, i, cv::Scalar(255), 1, 8);
//	}
//
//	cv::imshow("Contours Image", contoursImage); //轮廓  
//	cv::imshow("Point of Contours", Contours);   //向量contours内保存的所有轮廓点集 
//}
//
////局部特征点进行匹配,进行子图像提取
//void doMatch(Mat &image0, Mat &image2)
//{
//	cvtColor(image0, image0, COLOR_BGR2GRAY);
//	cv::Mat image1;
//	cv::resize(image0, image1, image1.size(), 0.5, 0.5, 1);
//	/*1.首先检测每幅图像的关键点*/
//		//创建存储关键点的容器
//	std::vector<cv::KeyPoint>keypoints1, keypoints2;
//	//定义特征检测器
//	cv::Ptr<cv::FeatureDetector>ptrDetector;   //泛型检测器指针
//	//这里选用FAST检测器
//	ptrDetector = cv::FastFeatureDetector::create(80);
//	//检测关键点
//	ptrDetector->detect(image1, keypoints1);
//	ptrDetector->detect(image2, keypoints2);
//
//	/*2.定义一个特定大小的矩形，用于表示每个关键点周围的图像块*/
//		//定义正方形的邻域
//	const int nsize = 11;         //邻域的尺寸
//	cv::Rect neighborhood(0, 0, nsize, nsize);  //11✖11
//	cv::Mat patch1;           //图像的块
//	cv::Mat patch2;
//
//	/*3.将一副图像的关键点与另一幅图像的全部关键点进行比较*/
//	/*并在第二幅图像中找出与第一幅图像中的每个关键点最相似的图像块*/
//		//在第二幅图像中找出与第一幅图像中的每个关键点最匹配的
//	cv::Mat result;
//	std::vector<cv::DMatch>matches;
//	//针对图像一中的全部关键点
//	for (int i = 0; i < keypoints1.size(); i++)
//	{
//		//定义图像块，目的是为了获得特征点所在的方形的左上角坐标
//		neighborhood.x = keypoints1[i].pt.x - nsize / 2;
//		neighborhood.y = keypoints1[i].pt.y - nsize / 2;
//
//		//如果邻域超出图像范围，继续处理下一个点
//		if (neighborhood.x < 0 || neighborhood.y < 0 ||
//			neighborhood.x + nsize >= image1.cols ||
//			neighborhood.y + nsize >= image1.rows)
//			continue;
//
//		//第一幅图像的块
//		patch1 = image1(neighborhood);
//
//		//存放最匹配的值
//		cv::DMatch bestMatch;
//
//		//针对第二幅图像的全部关键点
//		for (int j = 0; j < keypoints2.size(); j++)
//		{
//			//定义图像块
//			neighborhood.x = keypoints2[j].pt.x - nsize / 2;
//			neighborhood.y = keypoints2[j].pt.y - nsize / 2;
//
//			//如果邻域超出图像范围，就继续处理下一个点
//			if (neighborhood.x < 0 || neighborhood.y < 0 ||
//				neighborhood.x + nsize >= image2.cols ||
//				neighborhood.y + nsize >= image2.rows)
//				continue;
//
//			//第二幅图像的块
//			patch2 = image2(neighborhood);
//
//			//匹配两个图像块
//			cv::matchTemplate(patch1, patch2, result, cv::TM_SQDIFF);
//
//			//检测是否为最佳匹配
//			if (result.at<float>(0, 0) < bestMatch.distance)
//			{
//				bestMatch.distance = result.at<float>(0, 0);
//				bestMatch.queryIdx = i;
//				bestMatch.trainIdx = j;
//			}
//		}
//		//添加最佳匹配
//		matches.push_back(bestMatch);
//	}
//
//	/*4.根据相似度对特征点进行排序，并删除掉一些相似点*/
//	//提取25个最佳匹配项
//	std::nth_element(matches.begin(), matches.begin() + 25, matches.end());
//	matches.erase(matches.begin() + 25, matches.end());
//
//	/*5.把两幅图像拼接起来，然后用线条连接每个对应的点*/
//	//画出匹配结果
//	cv::Mat matchImage;
//	cv::drawMatches(image1, keypoints1, image2, keypoints2, matches,
//		matchImage, cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0));
//
//	cv::imshow("匹配图像", matchImage);
//}
//
///*1.获取每个视角螺纹所在子图像*/
//
////在文件中读取图像
//char * GetFileName(const char * dir, int i)
//{
//
//	char *name = new char[100];
//	if (i < 10) {
//		sprintf(name, "%s\\000%d-001.jpg", dir, i);
//		return name;
//	}
//	else if (i >= 10) {
//		sprintf(name, "%s\\00%d-001.jpg", dir, i);
//		return name;
//	}
//
//
//}
//
////将输入的图片逆时针旋转
//void rotate(cv::Mat &image, cv::Mat &rectImage, int angle)
//{
//
//
//	float radian = (float)(0.5 * CV_PI);
//
//	//填充图像
//	int maxBorder = (int)(max(image.cols, image.rows)* 1.414); //即为sqrt(2)*max
//	int dx = (maxBorder - image.cols) / 2;
//	int dy = (maxBorder - image.rows) / 2;
//	copyMakeBorder(image, rectImage, dy, dy, dx, dx, BORDER_CONSTANT);
//	//旋转
//	Point2f center((float)(rectImage.cols / 2), (float)(rectImage.rows / 2));
//	Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);//求得旋转矩阵
//	warpAffine(rectImage, rectImage, affine_matrix, rectImage.size());
//	//计算图像旋转之后包含图像的最大的矩形
//	float sinVal = abs(sin(radian));
//	float cosVal = abs(cos(radian));
//	Size targetSize((int)(image.cols * cosVal + image.rows * sinVal),
//		(int)(image.cols * sinVal + image.rows * cosVal));
//
//	//剪掉多余边框
//	int x = (rectImage.cols - targetSize.width) / 2;
//	int y = (rectImage.rows - targetSize.height) / 2;
//	Rect rect(x, y, targetSize.width, targetSize.height);
//	rectImage = Mat(rectImage, rect);
//
//
//	//cv::imshow("原图像", image);
//	//cv::imshow("旋转图像", rectImage);
//}
//
////通过自定义框，选中螺纹区域，之后将子区域提取出来
//void getWhorl(cv::Mat &image, cv::Mat &rectImage, int x, int width)
//{
//
//	//将图像改为单通道灰度图
//	//cv::Mat grayImage(image.size(),image.type(),cv::Scalar(0));
//	//cv::cvtColor(image, grayImage, COLOR_BGR2GRAY);
//
//	//自定义矩形框信息，将螺纹区域的左右边界确定，切割出来
//
//
//	cv::Rect rect(x, 0, width, image.rows);
//	cv::Mat rectImage01 = image(rect);
//	//cv::Rect rect(360, 0, 168, image.rows);
//	//rectImage01 = image(rect);
//
//	//对已经切割的图像进行二值化处理
//	cv::Mat threImage;
//	cv::threshold(rectImage01, threImage, 200, 255, cv::THRESH_BINARY_INV);
//
//	//利用形态学方法对边角进行修整
//	cv::Mat erodedImage;
//	cv::erode(threImage, erodedImage, cv::Mat(), cv::Point(-1, -1), 8);		//腐蚀
//
//	//对其进行膨胀处理，将间隙填充
//	cv::Mat dilatedImage;
//	cv::dilate(erodedImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 1);
//	//通过连通域分析，将其框起来，即确定其上下边界
//	//定义轮廓数值
//	std::vector<std::vector<cv::Point>>contours;
//	cv::findContours(erodedImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
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
//	//切割
//	cv::Rect r0 = cv::boundingRect(contours[k]);
//	rectImage = rectImage01(r0);
//
//
//}
//
//void allWhorl(std::vector<cv::Mat>&vAllimg)
//{
//	string imagePath, imagePath1, imagePath2;
//
//	std::vector<cv::Mat>vImg, vImg1, vImg2;
//	//vImg.resize(40);
//	//vImg1.resize(40);
//	//vImg2.resize(40);
//	for (int i = 0; i < 59; i++) {
//		imagePath = GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹A检测\\合格\\0", i);
//		imagePath1 = GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹B检测\\合格\\0", i);
//		imagePath2 = GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹C检测\\合格\\0", i);
//		cv::Mat img = cv::imread(imagePath);
//		cv::Mat img1 = cv::imread(imagePath1);
//		cv::Mat img2 = cv::imread(imagePath2);
//		if (img.empty() || img1.empty() || img2.empty())
//		{
//			cout << "image open error" << endl;
//
//		}
//		cv::Mat gray, gray1, gray2;
//		cvtColor(img, gray, COLOR_BGR2GRAY);
//		cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
//		cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
//		vImg.push_back(gray);
//		vImg1.push_back(gray1);
//		vImg2.push_back(gray2);
//	}
//	//for (int i = 0; i < vImg.size(); i++)
//	//{
//	//	cv::imshow("", vImg[i]);
//	//	waitKey(2000);
//	//}
//	std::vector<cv::Mat>vRectImg, vRectImg1, vRectImg2;
//	//vRectImg.resize(40);
//	//vRectImg1.resize(40);
//	//vRectImg2.resize(40);
//	cv::Mat rectImg, rectImg1, rectImg2;
//	for (int i = 0; i < vImg.size(); i++)
//	{
//		//1.切割子图像
//		getWhorl(vImg[i], rectImg, 354, 168);
//		vRectImg.push_back(rectImg);
//		getWhorl(vImg1[i], rectImg1, 356, 168);
//		vRectImg1.push_back(rectImg1);
//		getWhorl(vImg2[i], rectImg2, 380, 168);
//		vRectImg2.push_back(rectImg2);
//	}
//
//	//vAllimg.resize(120);
//	for (int i = 0; i < vRectImg.size() - 1; i = i + 2)
//	{
//		vAllimg.push_back(vRectImg[i]);
//		vAllimg.push_back(vRectImg[i + 1]);
//
//		vAllimg.push_back(vRectImg2[i]);
//		vAllimg.push_back(vRectImg2[i + 1]);
//
//		vAllimg.push_back(vRectImg1[i]);
//		vAllimg.push_back(vRectImg1[i + 1]);
//
//	}
//	//for (int i = 0; i < 6; i++)
//	//{
//	//	cv::imshow(GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹A检测\\合格\\0", i), vAllimg[i]);
//	//	cv::waitKey(0);
//	//}
//
//}
//
///*2.获取螺旋区域直径，进行柱面展开*/
//
//void speardCylinder(cv::Mat &rectImage01, cv::Mat &speImage)
//{
//	cv::Mat rectImage;
//	rotate(rectImage01, rectImage, 270);
//	double radius = rectImage.cols / 2.0;		//定义半径，为子图像宽的一半
//	speImage.create(rectImage.rows, PI*radius, CV_8UC1);
//
//	//映射参数
//	cv::Mat mapX(speImage.size(), CV_32F);
//	cv::Mat mapY(speImage.size(), CV_32F);
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
//	cv::remap(rectImage, speImage, mapX, mapY, cv::INTER_LINEAR);
//	//	cv::imshow("重投影", speImage);
//}
//
///*3. 根据相邻两个区域之间夹角的初始值，在展开图重叠部分进行模板匹配，
//在瓶胚的竖直方向上匹配搜索的范围可以小一点，然后将同一相机的图像进行拼接；*/
//
///*3.1 通过模板匹配的方法求取平移变换参数*/
///**求平移量
// *参数表为输入两幅图像（有一定重叠区域）
// *返回值为点类型，存储x,y方向的偏移量
//*/
//
//
////利用模板匹配，进行图像拼接
//void getStitcherImage(cv::Mat &speImage01, cv::Mat &speImage02, cv::Mat& dstImg, int angle)
//{
//
//	int c = speImage02.cols;
//	//利用旋转角度，确定大概的拼接位置
//	int col = (c / 2 + (c*double(angle / 180.0) / 2));
//
//	cv::Rect rectCut(col, 0, 20, speImage02.rows);			//定义模板位置大小
//	cv::Rect rectMatched(c / 4, 0, c / 4, speImage02.rows);
//	cv::Mat imgTemp = speImage02(rectCut);			//在左图像上取模板
//	cv::Mat imgMatched = speImage01(rectMatched);			//取右图像左半区域进行模板匹配
//
//	int width = imgMatched.cols - imgTemp.cols + 1;
//	int height = imgMatched.rows - imgTemp.rows + 1;
//	cv::Mat matchResult(height, width, CV_32FC1);
//	cv::matchTemplate(imgMatched, imgTemp, matchResult, cv::TM_CCORR_NORMED);			//匹配
//	cv::normalize(matchResult, matchResult, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围
//
//	double minValue, maxValue;
//	cv::Point minLoc, maxLoc;
//	cv::minMaxLoc(matchResult, &minValue, &maxValue, &minLoc, &maxLoc);		//找到最相似的位置
//
//	//定义拼接图像
//	int newCol = maxLoc.x + c / 4;
//	cv::Mat dst(speImage01.rows, speImage01.cols + rectCut.x - newCol, CV_8UC1);
//	cv::Mat roiLeft = dst(Rect(0, 0, speImage02.cols, speImage02.rows));	//公共区域左部分
//	speImage02.copyTo(roiLeft);
//
//	//在有图上画出模板区域
//
//	cv::Mat rDebugImg = speImage01.clone();
//	cv::rectangle(rDebugImg, Rect(newCol, maxLoc.y, imgTemp.cols, imgTemp.rows), Scalar(255, 255, 0), 2, 8);
//	//在左图上画出模板区域
//	cv::Mat lDebugImg = speImage02.clone();
//	cv::rectangle(lDebugImg, Rect(col, 0, imgTemp.cols, imgTemp.rows), Scalar(255, 255, 0), 2, 8);
//
//	//拼接公共区域右半部分
//	cv::Mat roiMatched = speImage01(Rect(newCol, maxLoc.y - rectCut.y, speImage01.cols - newCol, speImage01.rows - 1 - (maxLoc.y - rectCut.y)));
//	cv::Mat roiRight = dst(Rect(rectCut.x, 0, roiMatched.cols, roiMatched.rows));
//	roiMatched.copyTo(roiRight);
//
//	//利用加权，进行图像融合处理
//	cv::Mat leftTempImg = imgTemp;
//	cv::Mat rightTempImg = speImage01(Rect(newCol, maxLoc.y, imgTemp.cols, imgTemp.rows));
//	cv::Mat mergeImg(imgTemp.size(), imgTemp.type(), cv::Scalar(0));		//融合图像
//	for (int y = 0; y < imgTemp.rows; y++)
//	{
//		for (int x = 0; x < 20; x++)
//		{
//			int leftPixel = leftTempImg.at<uchar>(y, x);
//			int rightPixel = rightTempImg.at<uchar>(y, x);
//			mergeImg.at<uchar>(y, x) = (1 - (0.05*x))*leftPixel
//				+ (0.05*x)*rightPixel;
//
//		}
//
//	}
//	//cv::addWeighted(leftTempImg, 0.5, rightTempImg, 0.5, 0, mergeImg);
//
//	cv::Mat roi = dst(Rect(rectCut.x, 0, imgTemp.cols, imgTemp.rows));
//	mergeImg.copyTo(roi);
//
//
//	dstImg.create(dst.size(), dst.type());
//	dstImg = dst.clone();
//	//	cv::imshow("融合之后", dstImg);
//
//		//2.
//	//int c = speImage02.cols;
//		////利用旋转角度，确定大概的拼接位置
//		//int col = (c / 2 + (c*double(angle / 180.0) / 2));
//
//		//cv::Rect rectCut(col, 0, 20, speImage02.rows);			//定义模板位置大小
//		//cv::Rect rectMatched((c/4), 0, (c / 2), speImage02.rows);
//		//cv::Mat imgTemp = speImage02(rectCut);			//在左图像上取模板
//		//cv::Mat imgMatched = speImage01(rectMatched);			//取右图像左半区域进行模板匹配
//
//		//int width = imgMatched.cols - imgTemp.cols + 1;
//		//int height = imgMatched.rows - imgTemp.rows + 1;
//		//cv::Mat matchResult(height, width, CV_32FC1);
//		//cv::matchTemplate(imgMatched, imgTemp, matchResult, cv::TM_SQDIFF_NORMED);			//匹配
//		//cv::normalize(matchResult, matchResult, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围
//
//		//double minValue, maxValue;
//		//cv::Point minLoc, maxLoc;
//		//cv::minMaxLoc(matchResult, &minValue, &maxValue, &minLoc, &maxLoc);		//找到最相似的位置
//
//		////定义拼接图像
//		//cv::Mat dst(speImage01.rows, speImage01.cols + rectCut.x - minLoc.x, CV_8UC1);
//		//cv::Mat roiLeft = dst(Rect(0, 0, speImage02.cols, speImage02.rows));	//公共区域左部分
//		//speImage02.copyTo(roiLeft);
//		////speImage02.colRange(0, col).copyTo(dst.colRange(0, col));
//		////在有图上画出模板区域
//		//cv::Mat debugImg = speImage01.clone();
//		//cv::rectangle(debugImg, Rect(minLoc.x, minLoc.y, imgTemp.cols, imgTemp.rows), Scalar(0, 255, 0), 2, 8);
//
//
//		//cv::Mat roiMatched = speImage01(Rect(minLoc.x, minLoc.y - rectCut.y, speImage01.cols - minLoc.x, speImage01.rows - 1 - (minLoc.y - rectCut.y)));
//		//cv::Mat roiRight = dst(Rect(rectCut.x, 0, roiMatched.cols, roiMatched.rows));
//		////speImage01.colRange(maxLoc.x, speImage01.cols).copyTo(dst.colRange(col, dst.cols));
//
//		//roiMatched.copyTo(roiRight);
//
//		////利用加权，进行图像融合处理
//		//cv::Mat leftTempImg = imgTemp;
//		//cv::Mat rightTempImg = speImage01(Rect(minLoc.x, minLoc.y, imgTemp.cols, imgTemp.rows));
//		//cv::Mat mergeImg(imgTemp.size(), imgTemp.type(), cv::Scalar(0));		//融合图像
//		//for (int y = 0; y < imgTemp.rows; y++)
//		//{
//		//	for (int x = 0; x < 20; x++)
//		//	{
//		//		int leftPixel = leftTempImg.at<uchar>(y, x);
//		//		int rightPixel = rightTempImg.at<uchar>(y, x);
//		//		mergeImg.at<uchar>(y, x) = (1 - (0.05*x))*leftPixel
//		//			+ (0.05*x)*rightPixel;
//
//		//	}
//
//		//}
//		////cv::addWeighted(leftTempImg, 0.5, rightTempImg, 0.5, 0, mergeImg);
//
//		//cv::Mat roi = dst(Rect(rectCut.x, 0, imgTemp.cols, imgTemp.rows));
//		//mergeImg.copyTo(roi);
//		////mergeImg.colRange(0, mergeImg.cols).copyTo(dst.colRange(col, col + imgTemp.cols));
//
//		//dstImg.create(dst.size(), dst.type());
//		//dstImg = dst.clone();
//		////	cv::imshow("融合之后", dstImg);
//
//
//}
//
////多张图像进行拼接融合
//void mulStitcherImage(std::vector<cv::Mat>&vSpeImg, cv::Mat &dst, int num, int *arr)
//{
//	cv::Mat temp = vSpeImg[0];//(vSpeImg[0].size(), vSpeImg[0].type());
//
//	//getStitcherImage(vSpeImg[0], vSpeImg[1], temp, angle);
//	std::vector<cv::Mat>vSticher;
//	vSticher.resize(6);
//
//	for (int i = 1; i < vSpeImg.size(); i++)
//	{
//
//		getStitcherImage(temp, vSpeImg[i], vSticher[i], arr[i - 1]);
//		//temp.create(dst.size(), dst.type());
//		temp = vSticher[i].clone();
//
//
//	}
//	dst = temp.clone();
//}
//
////通过转换后保存的图像，会失真,和imshow显示出的图像相差很大
//void writeImg(const char* filename, const cv::Mat& mat)
//{
//
//	cv::imwrite(filename, mat);
//}
//
////做一个首尾切割函数，确定周期，对拼接图像进行首尾切割，使其在一个周期
//void getPeriod(cv::Mat &img01, cv::Mat &img02, int angle, int &dis, cv::Point &p)
//{
//	//首先根据旋转角度，确定近似匹配位置
//	//在匹配左图的匹配位置列为
//	int col = (img01.cols / 2 + (img01.cols *double(angle / 180.0) / 2));
//	dis = img01.cols - col;			//此列到尾列的距离
//
//	//定义模板尺寸以及匹配位置
//	cv::Mat tempImg = img01(Rect(col, 0, 20, img01.rows));
//	cv::Mat matchImg = img02(Rect(0, 0, img02.cols / 2, img02.rows));
//
//	//模板匹配
//	cv::Mat result(matchImg.rows - tempImg.rows + 1, matchImg.cols - tempImg.cols + 1, CV_32FC1);
//	cv::matchTemplate(matchImg, tempImg, result, cv::TM_SQDIFF);
//
//	//找到最相似的位置
//	double minVal, maxVal;
//	cv::Point minPt, maxPt;
//	cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);
//	p = minPt;
//
//
//}
//
////对所有函数进行整合
//void resultImg(std::vector<cv::Mat>&vAllimg, int num, int *arr)
//{
//
//	std::vector<cv::Mat>vResult;
//	for (int j = 0; j < vAllimg.size(); j += 6)
//	{
//		//std::vector<cv::Mat>vRectImg;
//	//vRectImg.resize(num);
//		std::vector<cv::Mat>vSpeImg;
//		cv::Mat speImg;
//		for (int i = 0; i < num; i++)
//		{
//			//1.切割子图像
//			//getWhorl(vAllimg[i], vRectImg[i], 360, 168);
//		//	writeImg(GetFileName("D:\\files\\testImage", i), vRectImg[i]);
//			//cv::imshow(GetFileName("D:\\files\\testImage", i), vRectImg[i]);
//			//2.将切割出来的子图像进行柱体展开
//			speardCylinder(vAllimg[i + j], speImg);
//			vSpeImg.push_back(speImg);
//			//cv::imwrite(GetFileName("D:\\files\\testImage", i), vSpeImg[i]);
//
//		}
//		//3.拼接
//		cv::Mat dst;
//		mulStitcherImage(vSpeImg, dst, num, arr);
//		//	cv::imshow("拼接图像", dst);
//			//4.确定周期，对拼接图像进行首尾切割，使其在一个周期
//		cv::Point p;
//		int distance;
//		getPeriod(vSpeImg[0], vSpeImg[vSpeImg.size() - 1], 60, distance, p);
//		int weight = dst.cols - p.x - distance;
//		int height = dst.rows;
//		cv::Mat result = dst(Rect(p.x, 0, weight, height));
//		vResult.push_back(result);
//		//cv::imshow("结果图像", result);
//	}
//	//cv::imshow(" ",vResult[0]);
//	//cv::waitKey(0);
//	for (int i = 0; i < vResult.size(); i++)
//	{
//		//cv::imshow("", vResult[i]);
//		cv::imshow(GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹A检测\\合格\\0", i), vResult[i]);
//		cv::waitKey(1000);
//	}
//}
//
//void changeSize(cv::Mat &Tempimg, std::vector<cv::Mat>vImg, int num, int *arr)
//{
//	std::vector<cv::Mat>vResult;
//	for (int j = 0; j < vImg.size(); j += 6)
//	{
//		std::vector<cv::Mat>vResize, vGray, vRotate, vSpeImg;
//		vResize.resize(num);
//		vGray.resize(num);
//		vSpeImg.resize(num);
//		vRotate.resize(num);
//		for (int i = 0; i < num; i++)
//		{
//			cv::resize(vImg[i + j], vResize[i], Tempimg.size());
//			cv::cvtColor(vResize[i], vGray[i], cv::COLOR_BGR2GRAY);
//			//cv::imshow(GetFileName("D:\\files\\testImage", i), vGray[i]);
//
//			//2.将切割出来的子图像进行柱体展开
//			rotate(vGray[i], vRotate[i], 90);
//			speardCylinder(vRotate[i], vSpeImg[i]);
//			//	cv::imshow(GetFileName("D:\\files\\testImage", i), vSpeImg[i]);
//		}
//		//3.拼接
//		cv::Mat dst;
//		mulStitcherImage(vSpeImg, dst, num, arr);
//		//	cv::imshow("拼接图像", dst);
//
//			//4.确定周期，对拼接图像进行首尾切割，使其在一个周期
//		cv::Point p;
//		int distance;
//		getPeriod(vSpeImg[0], vSpeImg[vSpeImg.size() - 1], 60, distance, p);
//		int weight = dst.cols - p.x - distance;
//		int height = dst.rows;
//		cv::Mat result = dst(Rect(p.x, 0, weight, height));
//		vResult.push_back(result);
//		cv::imshow(GetFileName("D:\\files\\testImage\\螺纹采集图像\\正常\\螺纹A检测\\合格\\0", j), result);
//		//cv::waitKey(1000);
//	}
//
//}
//
//
////将所有拼接图像进行调整到相同周期下（开始、结束相同）
//void adjPeriod(cv::Mat &target, std::vector<cv::Mat>&img, std::vector<cv::Mat>&adjImg)
//{
//	target = img[0];
//	cv::Mat tempImg = target(Rect(0, 0, 20, target.rows));	//取出模板图像，为后面做匹配
//
//	for (int i = 1; i < img.size(); i++)
//	{
//		int weight = img[i].cols - 20 + 1;
//		int height = img[i].rows - target.rows + 1;
//		cv::Mat result(height, weight, CV_32FC1);
//
//		//进行模板匹配
//		cv::matchTemplate(img[i], tempImg, result, cv::TM_SQDIFF);
//
//		//找到最相似位置
//		double minVal, maxVal;
//		cv::Point minPt, maxPt;
//		cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);
//
//		//对图像周期进行调整，将其进行切割拼接
//		adjImg[i].create(img[i].rows, img[i].cols - 20, CV_8UC1);
//		cv::Mat roiRight = img[i](Rect(minPt.x, minPt.y, img[i].cols - minPt.x + 1 - 20, img[i].rows));
//		roiRight.copyTo(adjImg[i](Rect(0, 0, adjImg[i].cols - minPt.x - 1, adjImg[i].rows)));
//
//		cv::Mat roiLeft = img[i](Rect(0, 0, minPt.x + 1, img[i].rows));
//		roiLeft.copyTo(adjImg[i](Rect(adjImg[i].cols - minPt.x - 1, 0, minPt.x + 1, img[i].rows)));
//
//		//对拼接部分进行融合
//		cv::Mat leftEnd = img[i](Rect(img[i].cols - 10 - 1, 0, 10, img[i].rows));
//		cv::Mat rightHead = img[i](Rect(0, 0, 10, img[i].rows));
//		cv::Mat middle(leftEnd.size(), leftEnd.type(), cv::Scalar(0));
//		for (int y = 0; y < 10; y++)
//		{
//			for (int x = 0; x < 10; x++)
//			{
//				middle.at<uchar>(x, y) = (1 - (x * 0.1))*leftEnd.at<uchar>(x, y)
//					+ x * rightHead.at<uchar>(x, y);
//			}
//		}
//		middle.copyTo(adjImg[i](Rect(adjImg[i].cols - minPt.x - 1, 0, 10, adjImg[i].rows)));
//	}
//
//}