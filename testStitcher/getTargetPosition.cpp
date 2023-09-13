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

// �Զ��������Ƚ�����
bool myCompare(C_Info &v1, C_Info &v2) {
	return v1.centerScore > v2.centerScore;
}

//����mask
cv::Mat getTargetPosition::doMask(cv::Mat& img)
{
	//��ֵ��
	cv::Mat binImg;
	cv::threshold(img, binImg, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
	//return binImg;
	//��ͼ����п���������ڲ��׶�
	cv::Mat closeImg;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
	cv::morphologyEx(binImg, closeImg, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);

	return closeImg;
}

//��ͨ���������ȡĿ��
std::vector<cv::Mat> getTargetPosition::connect_get_target(cv::Mat& targetImg,double singleArea,double areaThre)
{

	//�����������������
//cv::Mat element_01 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
//cv::morphologyEx(filledImg, filledImg, cv::MORPH_OPEN, element_01, cv::Point(-1, -1), 2);


////��ͨ�������ȥ���������
//	cv::Mat labels, stats, centroids;
//	int numLabels = cv::connectedComponentsWithStats(filledImg, labels, stats, centroids);
//	//������ͨ�������СĿ����й���
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



	//��ͨ������ж��Ƿ����Ŀ�����С���趨����ֵ
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
			//��ʴ��������Ŀ��֮������ӽ���ϸ������
			cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
			cv::erode(targetImg, targetImg, kernel_erode);
			//������������Ŀ��֮������ӽ��ж���
			cv::Mat element_01 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
			cv::morphologyEx(targetImg, targetImg, cv::MORPH_OPEN, element_01, cv::Point(-1, -1), 1);
		}

	}
	return vTargetImg;


}

//����ǰ�����������λ��
cv::Point getTargetPosition::getCenter(cv::Mat& img)
{
	cv::Mat binImg;
	cv::threshold(img, binImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	//��������
	std::vector<std::vector<cv::Point>>contours;
	cv::findContours(binImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	//����ǰ�����������λ��
	cv::Moments moments = cv::moments(contours[0]);
	cv::Point center = cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
	return center;
}
//ȥ����ֵͼ��ı�������
cv::Mat getTargetPosition::eliminate_bin_background(cv::Mat target, cv::Mat mask,double numsThre)
{
	//��ֵ��
	cv::Mat binImg;
	cv::threshold(target, binImg, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
	//cv::absdiff(binImg, mask, binImg);
	//cv::absdiff(diffImage, cv::Scalar::all(0), diffImage);
	////�����������������
	//cv::Mat element_01 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	//cv::morphologyEx(binImg, binImg, cv::MORPH_OPEN, element_01, cv::Point(-1, -1), 2);

	//��ͼ����б�������б�Ե����
	cv::Mat closeImg;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));
	cv::morphologyEx(binImg, closeImg, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);

	//����������Դ�����ֵ�����������ڲ����
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

//ȥ���Ҷ�ͼ��ı�������
cv::Mat getTargetPosition::eliminate_gray_background(cv::Mat target, cv::Mat mask)
{
	//�Ҷ�ͼ��֮������
	cv::Mat target_blur, mask_blur;
	cv::medianBlur(target, target_blur, 9);
	cv::medianBlur(mask, mask_blur, 9);
	cv::absdiff(target_blur, mask_blur, target_blur);
	cv::medianBlur(target_blur, target_blur, 7);

	//��ֵ������
	cv::Mat binImg;
	cv::threshold(target_blur, binImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	//�����㣬��ͼ����Ѵ��Լ��ڲ��������
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
	cv::Mat closeImg;
	cv::morphologyEx(binImg, closeImg, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 3);

	return closeImg;
}

//ȥ��ͼ��ı�������
cv::Mat getTargetPosition::eliminate_background(cv::Mat target, cv::Mat mask,  double numsThre)
{
	cv::Mat maskBin = doMask(mask);

	//��ȡ�ҶȲ�ֵ������ֵ
	cv::Mat subGrayImg = eliminate_gray_background(target, mask);

	//��ȡ��ֵ��ֵ������ֵ
	cv::Mat subBinImg = eliminate_bin_background(target, maskBin, numsThre);

	cv::Mat resultImg;
	bitwise_or(subBinImg, subGrayImg, resultImg);

	return resultImg;
}

//��ͨ�������������������˵���ֻ����Ŀ��ͼ��
cv::Mat getTargetPosition::setTarget(cv::Mat &filledImg, double  areaThre)
{
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::erode(filledImg, filledImg, element, cv::Point(-1, -1), 3);
	//��ͨ�������ȥ���������
	cv::Mat labels, stats, centroids;
	int numLabels = cv::connectedComponentsWithStats(filledImg, labels, stats, centroids);
	//������ͨ�������СĿ����й���
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

	//��ʴ������ȥ���߽�������
	//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::erode(targetImg, targetImg, element, cv::Point(-1, -1), 3);
	return targetImg;
}

//����Ŀ��ͼ��Ľ��ؾ���
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

//���ýǵ��ⷽ���Խ��ؾ���ͼ����м��ǵ�
cv::Mat getTargetPosition::detectCorner(cv::Mat &distImg,double blockSize,double ksize, double cornerRatio)
{
	cv::Mat corners;
	cv::cornerHarris(distImg, corners, 2, ksize, 0.04);

	//�ǵ����
	double maxVal = 0.0;
	cv::minMaxLoc(corners, nullptr, &maxVal);
	double threshold = cornerRatio *maxVal;
	cv::Mat cornersThreshold;
	cv::threshold(corners, cornersThreshold, threshold, 255, cv::THRESH_TOZERO);

	// ��һ����Ӧֵ
	cv::Mat normalizedCorners;
	cv::normalize(cornersThreshold, normalizedCorners, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	cv::convertScaleAbs(normalizedCorners, normalizedCorners);

	//�����ǵ�
	cv::Mat drawCorner = distImg.clone();
	cv::cvtColor(drawCorner, drawCorner, COLOR_GRAY2BGR);
	cv::Mat drawCorner_01 = drawCorner.clone();

		// ���ƽǵ�
	for (int y = 0; y < normalizedCorners.rows; y++) {
		for (int x = 0; x < normalizedCorners.cols; x++) {
			if (normalizedCorners.at<uchar>(y, x) > 0) {
				cv::circle(drawCorner, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), 2, 1, 0);
			}
		}
	}

	return normalizedCorners;

}

//�Ǽ���ֵ���Ʒ����Խǵ����ɸѡ
std::vector<NMS::CornerPoint> getTargetPosition::NMSCornerPoint(cv::Mat &normalizedCorners, double nmsThre)
{
	//�Ǽ���ֵ���ƹ��˽ǵ�
	std::vector<NMS::CornerPoint>vCorPoints;
	for (int r = 0; r < normalizedCorners.rows; r++)
	{
		unsigned char* normalizedCorners_ptr = normalizedCorners.ptr<uchar>(r);
		for (int c = 0; c < normalizedCorners.cols; c++)
		{
			//������ֵ���սǵ���Ӧֵ�Խǵ���й���
			if (normalizedCorners_ptr[c] > 0)
			{
				NMS::CornerPoint CorPoint;
				CorPoint.location = cv::Point2f(c, r);
				CorPoint.response = normalizedCorners_ptr[c];
				vCorPoints.push_back(CorPoint);
			}
		}
	}
	//������Ӧֵ���н�������
	std::sort(vCorPoints.begin(), vCorPoints.end(), NMS::CompareResponse);
	//ִ�зǼ���ֵ����
	//float threshold = 80.0;			//����������ֵ
	std::vector<NMS::CornerPoint> filteredCorPoints = NMS::PerformNMS(vCorPoints, static_cast<float>(nmsThre));

	return filteredCorPoints;
	////�����ƺ�Ľǵ���ͼ���Ͻ��л���
	//for (int i = 0; i < filteredCorPoints.size(); i++)
	//{
	//	cv::circle(drawCorner_01, filteredCorPoints[i].location, 3, cv::Scalar(0, 0, 255), 2, 1, 0);
	//}
}

//��������֮��ľ���
double getTargetPosition::calculateDistance(const cv::Point& point1, const cv::Point& point2) {
	int dx = point2.x - point1.x;
	int dy = point2.y - point1.y;
	return std::sqrt(dx * dx + dy * dy);
}

//���㱳����ĳ�㵽ǰ����Ե�Ͼ�������ĵ�
cv::Point getTargetPosition::calcuMinDist_of_edge(cv::Point2f &pt, cv::Mat &img)
{
	// ��ȡǰ����Ե
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	// ����������λ��
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

//���ݽǵ��ȡ���ؾ��������λ��
std::vector<C_Info> getTargetPosition::getCornerCenter(cv::Mat &distImg, std::vector<NMS::CornerPoint> filteredCornerPoints)
{
	//// ��ȡ�ǵ㷽��
	//cv::Mat dx, dy;
	//cv::Sobel(distImg, dx, CV_32F, 1, 0);
	//cv::Sobel(distImg, dy, CV_32F, 0, 1);
	//cv::Mat angle;
	//cv::phase(dx, dy, angle, true);

	//// ��ʾ�ǵ�ͷ���
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

	////��ͨ�������ȡĿ�����
	//   // ������ͨ�������
	//cv::Mat labels, stats, centroids;
	//int numLabels = cv::connectedComponentsWithStats(distImg, labels, stats, centroids);

	//// ����ǰ��Ŀ������
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

	//����Ŀ������
	double wholeArea = static_cast<double>(cv::countNonZero(distImg));
	double erodeArea = wholeArea / 2;
	cv::Mat singleDist = distImg.clone();
	while (wholeArea > erodeArea)
	{	
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::erode(singleDist, singleDist, element);
		//����ǰ��ͼ������
		wholeArea = static_cast<double>(cv::countNonZero(singleDist));
	}
	//cv::Mat drawImg = singleDist;
	//cv::cvtColor(drawImg, drawImg, COLOR_GRAY2BGR);
	//����ǵ���ǰ����������ĵ�
	std::vector<C_Info> vC_infos;
	for (int i = 0; i < filteredCornerPoints.size(); i++)
	{
		//���㱳����ĳ�㵽ǰ����Ե�Ͼ�������ĵ�
		cv::Point center = calcuMinDist_of_edge(filteredCornerPoints[i].location, singleDist);
		//circle(drawImg, center, 2, cv::Scalar(0, 0, 255), -1, 8);
		C_Info c_info;
		c_info.centerPt = center;
		c_info.centerScore = filteredCornerPoints[i].response;
		vC_infos.push_back(c_info);
	}

	return vC_infos;


	////ͨ����ʴ�ķ�����Ŀ���������
	//for (int i = 0; i < vArea.size(); i++)
	//{
	//	double wholeArea = vArea[i];
	//	double halfArea = wholeArea / 2;
	//	while (wholeArea > halfArea)
	//	{
	//		cv::Mat singleDist = vDistImg[i];
	//		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
	//		cv::erode(singleDist, singleDist, element);
	//		//����ǰ��ͼ������
	//		wholeArea = static_cast<double>(cv::countNonZero(singleDist));
	//	}

	//	//����ǵ���ǰ����������ĵ�
	//	for (int i = 0; i < filteredCornerPoints.size(); i++)
	//	{
	//		//���㱳����ĳ�㵽ǰ����Ե�Ͼ�������ĵ�
	//		cv::Point center = calcuMinDist_of_edge(filteredCornerPoints[i].location, singleDist);
	//	}

	//}

	


}

//�����ĵ�λ�ð��սǵ���Ӧǿ�Ƚ������򣬻�ȡ���յ�ץȡ˳��
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

	//���սǵ���Ӧֵ�����ĵ��������
		// ʹ���Զ���ıȽϺ�����������������
	sort(vC_infos.begin(), vC_infos.end(), myCompare);

	return vC_infos;
}

//��ԭͼ���ϻ�������λ�ò����ݵ���Ϣǿ�Ƚ��б��
void getTargetPosition::drawTargetInfo(cv::Mat &drawImg,std::vector<C_Info>vC_infos)
{
	//�������ĵ�λ��,�����
	for (int i = 0; i < vC_infos.size(); i++)
	{
		cv::circle(drawImg, vC_infos[i].centerPt, 8, cv::Scalar(255, 0, 0), -1, 8);
		//��ͼ���ϻ������ֱ��
		cv::putText(drawImg, std::to_string(i+1), vC_infos[i].centerPt, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
	}
}

//����ֱ�߼��
void getTargetPosition::hough_detect_line(cv::Mat &image,cv::Mat mask,cv::Mat filledImg)
{
	// Ѱ������
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(filledImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// �������������Ŀհ�ͼ��
	cv::Mat contourImage = cv::Mat::zeros(filledImg.size(), CV_8UC3);

	// ��������
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(255, 0, 255); // ��ɫ
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
	//�Զ�����ͼ����б߽�ƽ������ʹ����̬ѧ����
	cv::bitwise_not(binImg, binImg, cv::Mat());
	cv::Mat openImg;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(binImg, openImg, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
	cv::Mat edge;
	cv::Canny(openImg, edge, 30,90);		//��Ե���

	std::vector<cv::Vec2f> lines;		//�洢��⵽��ֱ��

	//����ֱ�߼��
	cv::HoughLines(edge, lines, 1, CV_PI / 180, 80);

	//ֱ������


	// ��ͼ���ϻ��Ƽ�⵽��ֱ��
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
	////����ת��
	// cv::Mat img=cvarrToMat(IplImg);
	// cv::Mat toMaskImg=cvarrToMat(toMaskIplImg);
	// cv::Mat drawImg=cvarrToMat(drawIplImg);
	
	//��MASKͼ����лҶȻ�
	cv::Mat grayMask;
	if (toMaskImg.channels() == 3) {
		cv::Mat hsvMaskImg;
		std::vector<cv::Mat>channelsMaskImg;
		cv::cvtColor(toMaskImg, hsvMaskImg, COLOR_BGR2HSV);
		split(hsvMaskImg, channelsMaskImg);
		grayMask = channelsMaskImg[fParams[3]];
	}
	else grayMask = toMaskImg.clone();

	//��������Ҫ��mask
	cv::Mat mask = doMask(grayMask);

	//��ͼ����лҶȻ�
	cv::Mat grayImg;
	if (img.channels() == 3) {
		cv::Mat hsvImg;
		std::vector<cv::Mat>channelsImg;
		cv::cvtColor(img, hsvImg, COLOR_BGR2HSV);
		split(hsvImg, channelsImg);
		grayImg = channelsImg[fParams[3]];
	}
	else grayImg = img.clone();

	//ȥ��ͼ��ı�������
	cv::Mat filterImg=eliminate_background(grayImg, grayMask,fParams[0]);

	//��ͨ�������������������˵���ֻ����Ŀ��ͼ��
	cv::Mat targetImg = setTarget(filterImg, fParams[1]);

	//����Ŀ��ͼ��Ľ��ؾ���
	cv::Mat distImg = calcuDist(targetImg, fParams[4]);//fParams[4]  ���ؾ������ֵ

	//���ýǵ��ⷽ���Խ��ؾ���ͼ����м��ǵ�
	std::vector<cv::Mat>vSingleDistImgs;
	std::vector<cv::Mat>vSingleCornerImgs;
	   // ������ͨ�������
	cv::Mat labels, stats, centroids;
	int numLabels = cv::connectedComponentsWithStats(distImg, labels, stats, centroids);
	for (int label = 1; label < numLabels; ++label) {
		cv::Mat tmp = (labels == label);
		//cv::Mat targetImg;
		//targetImg.setTo(255, tmp);
		vSingleDistImgs.push_back(tmp);
		cv::Mat cornerImg = detectCorner(tmp, fParams[5], 3, fParams[6]);		//fParams[5],fParams[6]  ����ǵ�����Ҫ��blocksize�;���˳ߴ�
		vSingleCornerImgs.push_back(cornerImg);
	}
	

	//�Ǽ���ֵ���ƹ��˽ǵ�
	//���ݽǵ��ȡ���ؾ��������λ��
	std::vector<std::vector<C_Info>>vC_Infos;
	std::vector<C_Info> vC_result_infos;		//�洢���յĶ�λ��Ϣ
	cv::Mat drawCorner_01 = distImg.clone();
	cv::cvtColor(drawCorner_01, drawCorner_01, COLOR_GRAY2BGR);
	for (int i = 0; i < vSingleCornerImgs.size(); i++)
	{
		std::vector<C_Info>vC_infos;
		std::vector<NMS::CornerPoint> filteredCornerPoints = NMSCornerPoint(vSingleCornerImgs[i],  fParams[7]);	//fParams[7],fParams[8] �Խǵ���й��˵���ֵ���Լ��Ǽ���ֵ���Ƶİ뾶
		//�����ƺ�Ľǵ���ͼ���Ͻ��л���

		//for (int i = 0; i < filteredCornerPoints.size(); i++)
		//{
		//	cv::circle(drawCorner_01, filteredCornerPoints[i].location, 3, cv::Scalar(0, 0, 255), 2, 1, 0);
		//}

		//���ݽǵ��ȡ���ؾ��������λ��
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

	//�����ĵ�λ�ð��սǵ���Ӧǿ�Ƚ������򣬻�ȡ���յ�ץȡ˳��
	std::vector<C_Info> vC_infos = getSortCenter(vC_Infos);
	for (int i = 0; i < vC_infos.size(); i++)
	{
		vC_result_infos.push_back(vC_infos[i]);
		
	}

	//�洢Ŀ������ĵ�����
	for (int i = 0; i < vC_result_infos.size(); i++)
	{
		vTargetPos.push_back(vC_result_infos[i].centerPt);
	}

	//��ԭͼ���ϻ�������λ�ò����ݵ���Ϣǿ�Ƚ��б��
	drawTargetInfo(drawImg, vC_result_infos);

	////���ֱ��
	hough_detect_line(grayImg,grayMask, targetImg);

	////�����ת����ȥ
	//if (drawIplImg)
	//{
	//	cvCopy(&(IplImage)drawImg, drawIplImg);
	//}
}
//
//int main()
//{
//	string pattern_img = "test\\�ϰ���λ\\�ϰ���λ1\\���ϸ�\\0";
//	std::vector<std::string>vFiles;
//	cv::glob(pattern_img, vFiles, false);
//	if (vFiles.size() == 0) {
//		cout << "the input data is null" << endl;
//		return 2;
//	}
//
//	else {
//
//		//ѡȡmaskͼ��
//		double fParams[8] = { 400,4000,10000,1,70,2,0.1,80 };
//		cv::Mat maskImg = cv::imread(vFiles.at(1));
//
//		//����������иֻ�����м�����
//		cv::Rect rect = cv::Rect(maskImg.cols / 4, 0, maskImg.cols / 2, maskImg.rows);
//		cv::Mat rectMaskImg = maskImg(rect);
//
//		for (int i = 13; i < vFiles.size(); i++)
//		{
//			//������ԭ���洢����λ��
//			std::vector<cv::Point>vCenterPt;
//
//			//����ͼ��
//			cv::Mat inputImg = cv::imread(vFiles.at(i));
//			cv::Mat drawImg = inputImg.clone();
//
//			//����������иֻ�����м�����
//			cv::Rect rect = cv::Rect(inputImg.cols / 4, 0, inputImg.cols / 2, inputImg.rows);
//			cv::Mat rectImg = inputImg(rect);
//			cv::Mat rectDrawImg = drawImg(rect);
//			////��ˮ�뷽���ָ�Ŀ��
//			//waterThresh(rectImg,400,mask,4000);
//			//��ͨ�����������ȡĿ��λ��
//			getTargetPosition getTP;
//			getTP.targetPosition(rectImg, rectMaskImg,fParams, vCenterPt, rectDrawImg);
//
//		}
//	}
//	system("pause");
//	return 0;
//}