#include"getTargetPosition.h"

namespace NMS {
	//struct CornerPoint {
	//	cv::Point2f location;
	//	float response;
	//};

	bool CompareResponse(const CornerPoint& a, const CornerPoint& b) {
		return a.response > b.response;
	}

	std::vector<CornerPoint> PerformNMS(const std::vector<CornerPoint>& cornerPoints, float threshold) {
		std::vector<CornerPoint> filteredPoints;

		for (const auto& point : cornerPoints) {
			bool isLocalMax = true;

			for (const auto& neighbor : filteredPoints) {
				if (cv::norm(point.location - neighbor.location) < threshold) {
					isLocalMax = false;
					break;
				}
			}

			if (isLocalMax) {
				filteredPoints.push_back(point);
			}
		}

		return filteredPoints;
	}
}

// 自定义容器比较类型
bool myCompare(C_Info &v1, C_Info &v2) {
	return v1.centerScore > v2.centerScore;
}

//制作mask
cv::Mat getTargetPosition::doMask(cv::Mat& img)
{
	//二值化
	cv::Mat binImg;
	cv::threshold(img, binImg, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
	//return binImg;
	//对图像进行开运算计算内部孔洞
	cv::Mat closeImg;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
	cv::morphologyEx(binImg, closeImg, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);

	return closeImg;
}

//连通域分析来获取目标
std::vector<cv::Mat> getTargetPosition::connect_get_target(cv::Mat& targetImg,double singleArea,double areaThre)
{

	//开运算消除多余噪点
//cv::Mat element_01 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
//cv::morphologyEx(filledImg, filledImg, cv::MORPH_OPEN, element_01, cv::Point(-1, -1), 2);


////连通域分析，去掉干扰情况
//	cv::Mat labels, stats, centroids;
//	int numLabels = cv::connectedComponentsWithStats(filledImg, labels, stats, centroids);
//	//利用连通域面积对小目标进行过滤
//	cv::Mat targetImg = cv::Mat::zeros(filledImg.size(), CV_8UC1);
//	for (int i = 1; i < numLabels; ++i)
//	{
//		int area = stats.at<int>(i, cv::CC_STAT_AREA);
//		if (area > areaThre)
//		{
//			cv::Mat tmp = (labels == i);
//			targetImg.setTo(255, tmp);
//		}
//	}



	//连通域分析判断是否存在目标面积小于设定的阈值
	std::vector<cv::Mat>vTargetImg;
	while (true)
	{
		cv::Mat labels_01, stats_01, centroids_01;
		int numLabels_01 = cv::connectedComponentsWithStats(targetImg, labels_01, stats_01, centroids_01);

		int num = 0;
		for (int i = 1; i < numLabels_01; i++)
		{
			if (stats_01.at<int>(i, cv::CC_STAT_AREA) < singleArea)
			{
				num++;
				cv::Mat tmp_01 = (labels_01 == i);
				//cv::Mat tarImg;
				//tarImg.setTo(255, tmp_01);
				targetImg = targetImg - tmp_01;
				if (stats_01.at<int>(i, cv::CC_STAT_AREA) > areaThre / 2)
					vTargetImg.push_back(tmp_01);
			}
		}
		if (num == numLabels_01 - 1)
		{
			break;
		}
		else {
			//腐蚀操作，将目标之间的连接进行细化操作
			cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
			cv::erode(targetImg, targetImg, kernel_erode);
			//开运算消除将目标之间的连接进行断裂
			cv::Mat element_01 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
			cv::morphologyEx(targetImg, targetImg, cv::MORPH_OPEN, element_01, cv::Point(-1, -1), 1);
		}

	}
	return vTargetImg;


}

//计算前景区域的中心位置
cv::Point getTargetPosition::getCenter(cv::Mat& img)
{
	cv::Mat binImg;
	cv::threshold(img, binImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	//查找轮廓
	std::vector<std::vector<cv::Point>>contours;
	cv::findContours(binImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	//计算前景区域的中心位置
	cv::Moments moments = cv::moments(contours[0]);
	cv::Point center = cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
	return center;
}
//去除二值图像的背景干扰
cv::Mat getTargetPosition::eliminate_bin_background(cv::Mat target, cv::Mat mask,double numsThre)
{
	//二值化
	cv::Mat binImg;
	cv::threshold(target, binImg, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
	//cv::absdiff(binImg, mask, binImg);
	//cv::absdiff(diffImage, cv::Scalar::all(0), diffImage);
	////开运算消除多余噪点
	//cv::Mat element_01 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	//cv::morphologyEx(binImg, binImg, cv::MORPH_OPEN, element_01, cv::Point(-1, -1), 2);

	//对图像进行闭运算进行边缘连接
	cv::Mat closeImg;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));
	cv::morphologyEx(binImg, closeImg, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);

	//检测轮廓，对大于阈值的轮廓进行内部填充
	std::vector<std::vector<cv::Point>>vContours;
	cv::Mat filledImg = cv::Mat::zeros(target.size(), CV_8UC1);
	cv::findContours(closeImg, vContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	for (const auto& contour : vContours)
	{
		if (contour.size() > numsThre)
		{
			cv::drawContours(filledImg, vector<vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
		}
	}
	filledImg = filledImg - mask;

	return filledImg;
}

//去除灰度图像的背景干扰
cv::Mat getTargetPosition::eliminate_gray_background(cv::Mat target, cv::Mat mask)
{
	//灰度图像之间做差
	cv::Mat target_blur, mask_blur;
	cv::medianBlur(target, target_blur, 9);
	cv::medianBlur(mask, mask_blur, 9);
	cv::absdiff(target_blur, mask_blur, target_blur);
	cv::medianBlur(target_blur, target_blur, 7);

	//二值化操作
	cv::Mat binImg;
	cv::threshold(target_blur, binImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	//闭运算，将图像断裂处以及内部进行填充
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
	cv::Mat closeImg;
	cv::morphologyEx(binImg, closeImg, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 3);

	return closeImg;
}

//去除图像的背景干扰
cv::Mat getTargetPosition::eliminate_background(cv::Mat target, cv::Mat mask,  double numsThre)
{
	cv::Mat maskBin = doMask(mask);

	//获取灰度差值计算后的值
	cv::Mat subGrayImg = eliminate_gray_background(target, mask);

	//获取二值差值计算后的值
	cv::Mat subBinImg = eliminate_bin_background(target, maskBin, numsThre);

	cv::Mat resultImg;
	bitwise_or(subBinImg, subGrayImg, resultImg);

	return resultImg;
}

//连通域分析，将干扰情况过滤掉，只保留目标图像
cv::Mat getTargetPosition::setTarget(cv::Mat &filledImg, double  areaThre)
{
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::erode(filledImg, filledImg, element, cv::Point(-1, -1), 3);
	//连通域分析，去掉干扰情况
	cv::Mat labels, stats, centroids;
	int numLabels = cv::connectedComponentsWithStats(filledImg, labels, stats, centroids);
	//利用连通域面积对小目标进行过滤
	cv::Mat targetImg = cv::Mat::zeros(filledImg.size(), CV_8UC1);
	for (int i = 1; i < numLabels; ++i)
	{
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		if (area > areaThre)
		{
			cv::Mat tmp = (labels == i);
			targetImg.setTo(255, tmp);
		}
	}

	//腐蚀操作，去掉边界干扰情况
	//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::erode(targetImg, targetImg, element, cv::Point(-1, -1), 3);
	return targetImg;
}

//计算目标图像的近地距离
cv::Mat getTargetPosition::calcuDist(cv::Mat &targetImg, double distThre)
{
	cv::Mat dist_transform;
	cv::distanceTransform(targetImg, dist_transform, cv::DIST_L2, 3);

	cv::Mat unit_dist_transform;
	dist_transform.convertTo(unit_dist_transform, CV_8UC1);

	cv::Mat distImg;
	cv::threshold(unit_dist_transform, distImg, distThre, 255, cv::THRESH_BINARY);

	return distImg;
}

//利用角点检测方法对近地距离图像进行检测角点
cv::Mat getTargetPosition::detectCorner(cv::Mat &distImg,double blockSize,double ksize, double cornerRatio)
{
	cv::Mat corners;
	cv::cornerHarris(distImg, corners, 2, ksize, 0.04);

	//角点过滤
	double maxVal = 0.0;
	cv::minMaxLoc(corners, nullptr, &maxVal);
	double threshold = cornerRatio *maxVal;
	cv::Mat cornersThreshold;
	cv::threshold(corners, cornersThreshold, threshold, 255, cv::THRESH_TOZERO);

	// 归一化响应值
	cv::Mat normalizedCorners;
	cv::normalize(cornersThreshold, normalizedCorners, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	cv::convertScaleAbs(normalizedCorners, normalizedCorners);

	//画出角点
	cv::Mat drawCorner = distImg.clone();
	cv::cvtColor(drawCorner, drawCorner, COLOR_GRAY2BGR);
	cv::Mat drawCorner_01 = drawCorner.clone();

		// 绘制角点
	for (int y = 0; y < normalizedCorners.rows; y++) {
		for (int x = 0; x < normalizedCorners.cols; x++) {
			if (normalizedCorners.at<uchar>(y, x) > 0) {
				cv::circle(drawCorner, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), 2, 1, 0);
			}
		}
	}

	return normalizedCorners;

}

//非极大值抑制方法对角点进行筛选
std::vector<NMS::CornerPoint> getTargetPosition::NMSCornerPoint(cv::Mat &normalizedCorners, double nmsThre)
{
	//非极大值抑制过滤角点
	std::vector<NMS::CornerPoint>vCorPoints;
	for (int r = 0; r < normalizedCorners.rows; r++)
	{
		unsigned char* normalizedCorners_ptr = normalizedCorners.ptr<uchar>(r);
		for (int c = 0; c < normalizedCorners.cols; c++)
		{
			//设置阈值按照角点响应值对角点进行过滤
			if (normalizedCorners_ptr[c] > 0)
			{
				NMS::CornerPoint CorPoint;
				CorPoint.location = cv::Point2f(c, r);
				CorPoint.response = normalizedCorners_ptr[c];
				vCorPoints.push_back(CorPoint);
			}
		}
	}
	//按照响应值进行降序排序
	std::sort(vCorPoints.begin(), vCorPoints.end(), NMS::CompareResponse);
	//执行非极大值抑制
	//float threshold = 80.0;			//设置邻域阈值
	std::vector<NMS::CornerPoint> filteredCorPoints = NMS::PerformNMS(vCorPoints, static_cast<float>(nmsThre));

	return filteredCorPoints;
	////将抑制后的角点在图像上进行画出
	//for (int i = 0; i < filteredCorPoints.size(); i++)
	//{
	//	cv::circle(drawCorner_01, filteredCorPoints[i].location, 3, cv::Scalar(0, 0, 255), 2, 1, 0);
	//}
}

//计算两点之间的距离
double getTargetPosition::calculateDistance(const cv::Point& point1, const cv::Point& point2) {
	int dx = point2.x - point1.x;
	int dy = point2.y - point1.y;
	return std::sqrt(dx * dx + dy * dy);
}

//计算背景上某点到前景边缘上距离最近的点
cv::Point getTargetPosition::calcuMinDist_of_edge(cv::Point2f &pt, cv::Mat &img)
{
	// 提取前景边缘
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	// 查找最近点的位置
	double minDistance = std::numeric_limits<double>::max();
	cv::Point nearestPoint=cv::Point(0,0);

	for (const auto& contour : contours) {
		for (const auto& point : contour) {
			double distance = abs(calculateDistance(point, pt));
			if (distance < minDistance) {
				minDistance = distance;
				nearestPoint = point;
			}
		}
	}

	return nearestPoint;
}

//根据角点获取近地距离的中心位置
std::vector<C_Info> getTargetPosition::getCornerCenter(cv::Mat &distImg, std::vector<NMS::CornerPoint> filteredCornerPoints)
{
	//// 获取角点方向
	//cv::Mat dx, dy;
	//cv::Sobel(distImg, dx, CV_32F, 1, 0);
	//cv::Sobel(distImg, dy, CV_32F, 0, 1);
	//cv::Mat angle;
	//cv::phase(dx, dy, angle, true);

	//// 显示角点和方向
	//cv::Mat result;
	//cv::cvtColor(distImg, result, cv::COLOR_GRAY2BGR);

	//for (int i = 0; i < filteredCornerPoints.size(); i++)
	//{

	//	float angleRadians = angle.at<float>(filteredCornerPoints[i].location.y, filteredCornerPoints[i].location.x);
	//	cv::Point2f startPoint( filteredCornerPoints[i].location.x, filteredCornerPoints[i].location.y);
	//	cv::Point2f endPoint(filteredCornerPoints[i].location.x + std::cos(angleRadians)*distCenter,
	//		filteredCornerPoints[i].location.y + std::sin(angleRadians)*distCenter);

	//	//cv::line(result, startPoint, endPoint, cv::Scalar(0, 0, 255), 1);
	//	cv::arrowedLine(result, startPoint, endPoint, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
	//	cv::circle(result, startPoint, 2, cv::Scalar(0, 255, 0), -1);
	//}

	////连通域分析获取目标面积
	//   // 进行连通组件分析
	//cv::Mat labels, stats, centroids;
	//int numLabels = cv::connectedComponentsWithStats(distImg, labels, stats, centroids);

	//// 计算前景目标的面积
	//std::vector<double>vArea;
	//std::vector<cv::Mat>vDistImg;
	//for (int label = 1; label < numLabels; ++label) {
	//	double foregroundArea = stats.at<int>(label, cv::CC_STAT_AREA);
	//	vArea.push_back(foregroundArea);
	//	cv::Mat tmp = (labels == label);
	//	cv::Mat targetImg;
	//		targetImg.setTo(255, tmp);
	//	vDistImg.push_back(targetImg);
	//}

	//计算目标的面积
	double wholeArea = static_cast<double>(cv::countNonZero(distImg));
	double erodeArea = wholeArea / 2;
	cv::Mat singleDist = distImg.clone();
	while (wholeArea > erodeArea)
	{	
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::erode(singleDist, singleDist, element);
		//计算前景图像的面积
		wholeArea = static_cast<double>(cv::countNonZero(singleDist));
	}
	//cv::Mat drawImg = singleDist;
	//cv::cvtColor(drawImg, drawImg, COLOR_GRAY2BGR);
	//计算角点离前景区域最近的点
	std::vector<C_Info> vC_infos;
	for (int i = 0; i < filteredCornerPoints.size(); i++)
	{
		//计算背景上某点到前景边缘上距离最近的点
		cv::Point center = calcuMinDist_of_edge(filteredCornerPoints[i].location, singleDist);
		//circle(drawImg, center, 2, cv::Scalar(0, 0, 255), -1, 8);
		C_Info c_info;
		c_info.centerPt = center;
		c_info.centerScore = filteredCornerPoints[i].response;
		vC_infos.push_back(c_info);
	}

	return vC_infos;


	////通过腐蚀的方法对目标进行搜索
	//for (int i = 0; i < vArea.size(); i++)
	//{
	//	double wholeArea = vArea[i];
	//	double halfArea = wholeArea / 2;
	//	while (wholeArea > halfArea)
	//	{
	//		cv::Mat singleDist = vDistImg[i];
	//		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
	//		cv::erode(singleDist, singleDist, element);
	//		//计算前景图像的面积
	//		wholeArea = static_cast<double>(cv::countNonZero(singleDist));
	//	}

	//	//计算角点离前景区域最近的点
	//	for (int i = 0; i < filteredCornerPoints.size(); i++)
	//	{
	//		//计算背景上某点到前景边缘上距离最近的点
	//		cv::Point center = calcuMinDist_of_edge(filteredCornerPoints[i].location, singleDist);
	//	}

	//}

	


}

//对中心点位置按照角点响应强度进行排序，获取最终的抓取顺序
std::vector<C_Info> getTargetPosition::getSortCenter(std::vector<std::vector<C_Info>>vC_Infos)
{
	std::vector<C_Info>vC_infos;
	for (int i = 0; i < vC_Infos.size(); i++)
	{
		for (int j = 0; j < vC_Infos[i].size(); j++)
		{
			vC_infos.push_back(vC_Infos[i][j]);
		}
	}

	//按照角点响应值对中心点进行排序
		// 使用自定义的比较函数对向量进行排序
	sort(vC_infos.begin(), vC_infos.end(), myCompare);

	return vC_infos;
}

//在原图像上画出中心位置并根据点信息强度进行标号
void getTargetPosition::drawTargetInfo(cv::Mat &drawImg,std::vector<C_Info>vC_infos)
{
	//画出中心点位置,并标号
	for (int i = 0; i < vC_infos.size(); i++)
	{
		cv::circle(drawImg, vC_infos[i].centerPt, 8, cv::Scalar(255, 0, 0), -1, 8);
		//在图像上绘制数字标号
		cv::putText(drawImg, std::to_string(i+1), vC_infos[i].centerPt, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
	}
}

//霍夫直线检测
void getTargetPosition::hough_detect_line(cv::Mat &image,cv::Mat mask,cv::Mat filledImg)
{
	// 寻找轮廓
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(filledImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// 创建绘制轮廓的空白图像
	cv::Mat contourImage = cv::Mat::zeros(filledImg.size(), CV_8UC3);

	// 绘制轮廓
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(255, 0, 255); // 蓝色
		cv::drawContours(contourImage, contours, static_cast<int>(i), color, 2, cv::LINE_8, hierarchy);
	}

	cv::Mat blurImg,blurMask,subImg;
	//blur(image, blurImg, cv::Size(9, 9));
	//cv::medianBlur(image, blurImg, 5);
	//cv::medianBlur(mask, blurMask, 5);
	cv::absdiff(image, mask, subImg);
	//cv::blur(subImg, subImg, cv::Size(5, 5));
	cv::medianBlur(subImg, subImg, 15);

	cv::Mat binImg;
	cv::threshold(subImg, binImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	//对二进制图像进行边界平滑处理，使用形态学操作
	cv::bitwise_not(binImg, binImg, cv::Mat());
	cv::Mat openImg;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(binImg, openImg, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
	cv::Mat edge;
	cv::Canny(openImg, edge, 30,90);		//边缘检测

	std::vector<cv::Vec2f> lines;		//存储检测到的直线

	//霍夫直线检测
	cv::HoughLines(edge, lines, 1, CV_PI / 180, 80);

	//直线抑制


	// 在图像上绘制检测到的直线
	cv::Mat drawImg = edge.clone();
	cv::cvtColor(drawImg, drawImg, COLOR_GRAY2BGR);
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0];
		float theta = lines[i][1];
		double a = std::cos(theta);
		double b = std::sin(theta);
		double x0 = a * rho;
		double y0 = b * rho;
		cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
		cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
		cv::line(contourImage, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
	}
}


void getTargetPosition::targetPosition(cv::Mat& img, cv::Mat& toMaskImg, double* fParams,
	std::vector<cv::Point>& vTargetPos, cv::Mat& drawImg)
//void getTargetPosition::targetPosition(IplImage* IplImg, IplImage* toMaskIplImg, double* fParams,
//	std::vector<cv::Point>& vTargetPos, IplImage* drawIplImg)
{
	////类型转换
	// cv::Mat img=cvarrToMat(IplImg);
	// cv::Mat toMaskImg=cvarrToMat(toMaskIplImg);
	// cv::Mat drawImg=cvarrToMat(drawIplImg);
	
	//对MASK图像进行灰度化
	cv::Mat grayMask;
	if (toMaskImg.channels() == 3) {
		cv::Mat hsvMaskImg;
		std::vector<cv::Mat>channelsMaskImg;
		cv::cvtColor(toMaskImg, hsvMaskImg, COLOR_BGR2HSV);
		split(hsvMaskImg, channelsMaskImg);
		grayMask = channelsMaskImg[fParams[3]];
	}
	else grayMask = toMaskImg.clone();

	//制作所需要的mask
	cv::Mat mask = doMask(grayMask);

	//对图像进行灰度化
	cv::Mat grayImg;
	if (img.channels() == 3) {
		cv::Mat hsvImg;
		std::vector<cv::Mat>channelsImg;
		cv::cvtColor(img, hsvImg, COLOR_BGR2HSV);
		split(hsvImg, channelsImg);
		grayImg = channelsImg[fParams[3]];
	}
	else grayImg = img.clone();

	//去除图像的背景干扰
	cv::Mat filterImg=eliminate_background(grayImg, grayMask,fParams[0]);

	//连通域分析，将干扰情况过滤掉，只保留目标图像
	cv::Mat targetImg = setTarget(filterImg, fParams[1]);

	//计算目标图像的近地距离
	cv::Mat distImg = calcuDist(targetImg, fParams[4]);//fParams[4]  近地距离的阈值

	//利用角点检测方法对近地距离图像进行检测角点
	std::vector<cv::Mat>vSingleDistImgs;
	std::vector<cv::Mat>vSingleCornerImgs;
	   // 进行连通组件分析
	cv::Mat labels, stats, centroids;
	int numLabels = cv::connectedComponentsWithStats(distImg, labels, stats, centroids);
	for (int label = 1; label < numLabels; ++label) {
		cv::Mat tmp = (labels == label);
		//cv::Mat targetImg;
		//targetImg.setTo(255, tmp);
		vSingleDistImgs.push_back(tmp);
		cv::Mat cornerImg = detectCorner(tmp, fParams[5], 3, fParams[6]);		//fParams[5],fParams[6]  计算角点所需要的blocksize和卷积核尺寸
		vSingleCornerImgs.push_back(cornerImg);
	}
	

	//非极大值抑制过滤角点
	//根据角点获取近地距离的中心位置
	std::vector<std::vector<C_Info>>vC_Infos;
	std::vector<C_Info> vC_result_infos;		//存储最终的定位信息
	cv::Mat drawCorner_01 = distImg.clone();
	cv::cvtColor(drawCorner_01, drawCorner_01, COLOR_GRAY2BGR);
	for (int i = 0; i < vSingleCornerImgs.size(); i++)
	{
		std::vector<C_Info>vC_infos;
		std::vector<NMS::CornerPoint> filteredCornerPoints = NMSCornerPoint(vSingleCornerImgs[i],  fParams[7]);	//fParams[7],fParams[8] 对角点进行过滤的阈值，以及非极大值抑制的半径
		//将抑制后的角点在图像上进行画出

		//for (int i = 0; i < filteredCornerPoints.size(); i++)
		//{
		//	cv::circle(drawCorner_01, filteredCornerPoints[i].location, 3, cv::Scalar(0, 0, 255), 2, 1, 0);
		//}

		//根据角点获取近地距离的中心位置
		vC_infos=getCornerCenter(vSingleDistImgs[i], filteredCornerPoints);
		if (vC_infos.size() == 1)
		{
			vC_result_infos.push_back(vC_infos[0]);
		}
		else
		{
			vC_Infos.push_back(vC_infos);
		}
		
	}

	//对中心点位置按照角点响应强度进行排序，获取最终的抓取顺序
	std::vector<C_Info> vC_infos = getSortCenter(vC_Infos);
	for (int i = 0; i < vC_infos.size(); i++)
	{
		vC_result_infos.push_back(vC_infos[i]);
		
	}

	//存储目标的中心点坐标
	for (int i = 0; i < vC_result_infos.size(); i++)
	{
		vTargetPos.push_back(vC_result_infos[i].centerPt);
	}

	//在原图像上画出中心位置并根据点信息强度进行标号
	drawTargetInfo(drawImg, vC_result_infos);

	////检测直线
	hough_detect_line(grayImg,grayMask, targetImg);

	////将结果转换回去
	//if (drawIplImg)
	//{
	//	cvCopy(&(IplImage)drawImg, drawIplImg);
	//}
}
//
//int main()
//{
//	string pattern_img = "test\\断包定位\\断包定位1\\不合格\\0";
//	std::vector<std::string>vFiles;
//	cv::glob(pattern_img, vFiles, false);
//	if (vFiles.size() == 0) {
//		cout << "the input data is null" << endl;
//		return 2;
//	}
//
//	else {
//
//		//选取mask图像
//		double fParams[8] = { 400,4000,10000,1,70,2,0.1,80 };
//		cv::Mat maskImg = cv::imread(vFiles.at(1));
//
//		//对区域进行切割，只处理中间区域
//		cv::Rect rect = cv::Rect(maskImg.cols / 4, 0, maskImg.cols / 2, maskImg.rows);
//		cv::Mat rectMaskImg = maskImg(rect);
//
//		for (int i = 13; i < vFiles.size(); i++)
//		{
//			//参数：原来存储中心位置
//			std::vector<cv::Point>vCenterPt;
//
//			//读入图像
//			cv::Mat inputImg = cv::imread(vFiles.at(i));
//			cv::Mat drawImg = inputImg.clone();
//
//			//对区域进行切割，只处理中间区域
//			cv::Rect rect = cv::Rect(inputImg.cols / 4, 0, inputImg.cols / 2, inputImg.rows);
//			cv::Mat rectImg = inputImg(rect);
//			cv::Mat rectDrawImg = drawImg(rect);
//			////分水岭方法分割目标
//			//waterThresh(rectImg,400,mask,4000);
//			//连通域分析方法获取目标位置
//			getTargetPosition getTP;
//			getTP.targetPosition(rectImg, rectMaskImg,fParams, vCenterPt, rectDrawImg);
//
//		}
//	}
//	system("pause");
//	return 0;
//}