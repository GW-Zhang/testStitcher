//#include"Stitcher.h"
//
//
////���ļ��ж�ȡͼ��
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
////����ƴ�Ӻ���
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
//	int overallWidth = 0;					//�������һ���������ڵĿ��
//	int tempWidth = 0;
//	double start3 = (double)getTickCount();
//	getMostCorr(speardImg, resultImg, arrAngle, overallWidth, tempWidth);
//	double time3 = ((double)getTickCount() - start3) * 1000 / getTickFrequency();
//	cout << "time3:" << time3 << endl;
//	return resultImg;
//}
//
////1.ͼ�������������ɼ���ͼ�����Ʋ��ֿ��ܻ������б�����
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
////1.1 Ѱ�ҽ�����Ҫ��������
//void DoStitcher::findFeaturePoints(cv::Mat &img, int cutX, int width, cv::Point &pt1, cv::Point &pt2)
//{
//	//�����и�ͼ��ȥ�����߲��ֵ�Ӱ��
//	cv::Rect rect(cutX, 0, adjWidth, img.rows);
//	cv::Mat cutImg = img(rect);
//
//	//���и��ͼ����ж�ֵ������
//	cv::Mat threImg;
//	cv::threshold(cutImg, threImg, 200, 255, cv::THRESH_BINARY_INV);
//
//	//ͨ����ͨ��������������������ȷ�������±߽�
//	//����������ֵ
//	std::vector<std::vector<cv::Point>>contours;
//	cv::findContours(threImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//
//	//ɾ����Ч����,�ҳ��������
//	int max = contours[0].size();
//	int k = 0;
//	for (int i = 0; i < contours.size(); i++)
//	{
//		if (contours[i].size() >= max) {
//			max = contours[i].size();
//			k = i;
//		}
//	}
//	//��Ӿ���
//	cv::Rect r0 = cv::boundingRect(contours[k]);
//
//	//Ѱ��������
//	int topPointY = r0.tl().y;			//��������Y
//	int bottomPointY = r0.br().y;		//��������Y
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
////1.2 ��ͼ����нǶ�����
//void DoStitcher::adjustImg(cv::Mat &img, cv::Mat &adjImg, cv::Point pt1, cv::Point pt2)
//{
//	//��������x�����ƫ�ƣ�����ƽ��У��
//	img.copyTo(adjImg);
//	cv::Point center(((pt1.x + pt2.x) / 2), ((pt1.y + pt2.y) / 2));		//�������ߵ�����λ��
//	double length = (pt2.y - pt1.y) / 2;
//
//	double num = pt1.x - pt2.x;			//�����������
//
//	double moveNum = num / double(pt2.y - pt1.y);
//	//�������ĵ���ϰ벿��
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
//	//�������ĵ���°벿��
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
////2.ͼ���и���������������и��
////ͨ���Զ����ѡ����������֮����������ȡ����
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
//		//���Ѿ��и��ͼ����ж�ֵ������
//		cv::Mat threImage;
//		cv::threshold(rectImage01, threImage, 200, 255, cv::THRESH_BINARY_INV);
//
//		////������̬ѧ�����Ա߽ǽ�������
//		//cv::Mat erodedImage;
//		//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
//		//cv::erode(threImage, erodedImage, element, cv::Point(-1, -1), 5);		//��ʴ
//
//		////����������ʹ�������϶���
//		//cv::Mat dilatedImage;
//		//cv::dilate(erodedImage, dilatedImage, cv::Mat(), cv::Point(-1, -1), 1);
//		//ͨ����ͨ��������������������ȷ�������±߽�
//		//����������ֵ
//		std::vector<std::vector<cv::Point>>contours;
//		cv::findContours(threImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//
//		//ɾ����Ч����,�ҳ��������
//		int max = contours[0].size();
//		int k = 0;
//		for (int i = 0; i < contours.size(); i++)
//		{
//			if (contours[i].size() >= max) {
//				max = contours[i].size();
//				k = i;
//			}
//		}
//		//�и�
//		cv::Rect r0 = cv::boundingRect(contours[k]);
//		int centerY = r0.height / 2 + r0.y;
//		int leftBegin = centerY - 178;
//		//int leftBegin = r0.tl().y;
//		//cv::Rect r1(r0.x - 100 + x, r0.y, r0.width + 300, r0.height);
//		cv::Rect r1(r0.x - 55 + arrCutX[i/2], leftBegin, r0.width + 55, 356);
//		//cv::Rect r1(r0.x - 55 + x, leftBegin, r0.width + 55, r0.height);
//		//rectImage = rectImage01(r0);
//		cv::Mat rectImage02 = adjImg[i](r1);
//		//������ת����
//		cv::Mat temp;
//		transpose(rectImage02, temp);
//		flip(temp, rectImg[i], 1);
//		//cv::flip(rectImage02, rectImg[i], 1);
//		//rotate(rectImage02, rectImg[i], 270);
//	}
//}
//
////2.2 //�������ͼƬ��ʱ����ת
//void DoStitcher::rotate(cv::Mat &image, cv::Mat &rectImage, int angle)
//{
//	Point center(image.cols / 2, image.rows / 2); //��ת����
//	Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
//	warpAffine(image, rectImage, rotMat, Size(image.rows, image.cols));
//	
//	//float radian = (float)(0.5 * CV_PI);
//
//	////���ͼ��
//	//int maxBorder = (int)(max(image.cols, image.rows)* 1.414); //��Ϊsqrt(2)*max
//	//int dx = (maxBorder - image.cols) / 2;
//	//int dy = (maxBorder - image.rows) / 2;
//	//copyMakeBorder(image, rectImage, dy, dy, dx, dx, BORDER_CONSTANT);
//	////��ת
//	//Point2f center((float)(rectImage.cols / 2), (float)(rectImage.rows / 2));
//	//Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);//�����ת����
//	//warpAffine(rectImage, rectImage, affine_matrix, rectImage.size());
//	////����ͼ����ת֮�����ͼ������ľ���
//	//float sinVal = abs(sin(radian));
//	//float cosVal = abs(cos(radian));
//	//Size targetSize((int)(image.cols * cosVal + image.rows * sinVal),
//	//	(int)(image.cols * sinVal + image.rows * cosVal));
//
//	////��������߿�
//	//int x = (rectImage.cols - targetSize.width) / 2;
//	//int y = (rectImage.rows - targetSize.height) / 2;
//	//Rect rect(x, y, targetSize.width, targetSize.height);
//	//rectImage = Mat(rectImage, rect);
//
//}
//
////3.�����Ʋ��ֽ�������չ��
//void DoStitcher::speardCylinder(std::vector<cv::Mat>rectImg, std::vector<cv::Mat>&speardImg, int radius)
//{
//	//3.1 ��ȡչ������
//	cv::Mat mapX, mapY;
//	speardCylinderMatrix(rectImg[0], radius, mapX, mapY);
//	speardImg.resize(rectImg.size());
//#pragma omp parallel for
//	for (int i = 0; i < rectImg.size(); i++)
//	{
//		//������ͶӰ���󣬽�������չ��
//		cv::remap(rectImg[i], speardImg[i], mapX, mapY, cv::INTER_LINEAR);
//	}
//}
//
////3.1��ȡչ������
//void DoStitcher::speardCylinderMatrix(cv::Mat &rectImage01, int &radius, cv::Mat &mapX, cv::Mat &mapY)
//{
//
//	radius = rectImage01.cols / 2.0;		//����뾶��Ϊ��ͼ����һ��
//	cv::Mat speImage(rectImage01.rows, PI*radius, CV_8UC1);
//
//	//ӳ�����
//	mapX.create(speImage.size(), CV_32F);
//	mapY.create(speImage.size(), CV_32F);
//
//	//����ӳ�����
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
//	//	cv::imshow("��ͶӰ", speImage);
//}
//
////4.����ģ��ƥ�䣬��չ��ͼ�����ͼ��ƴ��
//void DoStitcher::getMostCorr(std::vector<cv::Mat>&vImg, cv::Mat &allImg, int *arr, int &overallWidth, int &tempWidth)
//{
//	//���������洢ģ��ƥ������Ļ���ص÷�ͼ
//	std::vector<cv::Mat>vMatchResult;
//	vMatchResult.resize(6);
////#pragma omp parallel for
//	for (int i = 0; i < vImg.size() - 1; i++)	//�洢ǰ����÷�ͼ
//	{
//		//��ȡƥ����ض�
//		getCorr(vImg[i], vImg[i + 1], vMatchResult[i], arr[i]);
//	}
//
//	getCorr(vImg[5], vImg[0], vMatchResult[5], arr[5]);		//�洢�������÷�ͼ
//
//
//	int c = vImg[0].cols;			//ͼ����
//	int arrX[6] = { 0 };			//�����������洢��ȷ��ת�Ƕ�����ͼ�ж�Ӧ��λ��
//	int sumX = 0;
////#pragma omp parallel for
//	for (int i = 0; i < 6; i++)
//	{
//		arrX[i] = c - (double(arr[i] / 180.0)*c + c) / 2;
//		sumX += arrX[i];
//	}
//	//sumX = arrX[0]+ arrX[1]+ arrX[2]+ arrX[3]+ arrX[4]+ arrX[5];			//������λ�ý����ۼӣ�����������ƫ����x֮���Ϊ0
//	int arrBestX[6] = { 0 };		//�����洢���ƥ����Xλ��
//	int arrBestY[6] = { 0 };		//�洢���ƥ����Yλ��
//	float maxScore = 0.0;
//
//	//ѭ��������������ͼ��ƫ����֮��Ϊ0��ǰ���£�Ѱ�ҳ�ƥ��÷����λ��
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
//							int arrPixelX[6] = { x1,x2,x3,x4,x5,x6 };		//������λ�ã��������飬�����������
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
//									//cv::minMaxLoc(matchImg, &minValue, &maxValue, &minLoc, &maxLoc);		//�ҵ���߷ֵ�λ��
//									//vPt.push_back(maxLoc);
//
//									//float maxValue = vMatchResult[i].at<float>(0, (arrPixelX[i] - (arrX[i] - 3) - 1));
//									float maxValue = 0.0;
//									int maxY = 0;																		//float pixel = 0.0;
//									for (int y = 0; y < vMatchResult[0].rows; y++)		//Ѱ��Y��������÷�λ��
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
//										Score += maxValue;		//���÷�
//									}
//								}
//
//								if (Score >= maxScore)		//Ѱ�ҳ������÷�
//								{
//									maxScore = Score;
//									for (int i = 0; i < 6; i++)
//									{
//										arrBestX[i] = arrPixelX[i];		//������λ����Ϣ
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
//	//����arrBestY�ĸ���
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
//	//�ͷ�����
//	std::vector<cv::Mat>().swap(vMatchResult);
//	//����ƴ��
//	cv::Mat dst;
//	cv::Mat temp;
//	StitImg(vImg[0], vImg[1], temp, arr[0], arrBestX[0], arrBestY[0]);
////#pragma omp parallel for
//	for (int i = 1; i < 5; i++)
//	{
//		StitImg(temp, vImg[i + 1], dst, arr[i], arrBestX[i], arrBestY[i]);
//		temp = dst;
//	}
//	//�ٽ���һ�������һ�Ž���ƴ�ӣ���֤ƴ����ͼ�����㹻�����࣬�����������
//	StitImg(temp, vImg[0], dst, arr[5], arrBestX[5], arrBestY[5]);
//	//������ͼ�������β�и�
//	////Ϊ����չ��ͼ��Ĵ�Լ���
//	//int col = (c / 2 + (c*double(arr[5] / 180.0) / 2));
//	allImg = dst(cv::Rect(c / 4, 0, dst.cols - c / 2, dst.rows));
//
//	overallWidth = allImg.cols - c / 2;		//һ����������ƴ�Ӻ�Ŀ��
//	tempWidth = c / 4;						//ѡȡģ��Ŀ�ȣ�Ϊ�������ڵ�����׼��
//
//
//}
//
////4.1 //��ȡģ��ƥ����ض�
//void DoStitcher::getCorr(cv::Mat &img1, cv::Mat &img2, cv::Mat &matchResult, int angle)
//{
//	int c = img2.cols;	//Ϊ����չ��ͼ��Ĵ�Լ���
//	int col = (c / 2 + (c*double(angle / 180.0) / 2));		//��ͼƥ�䴦
//	int rCol = c - col;
//
//	//cv::Rect rectCut(col, 0, 20, img2.rows);
//	cv::Rect rectCut(col, 40, 20, img2.rows - 45);			//����ģ��λ�ô�С
//	cv::Rect rectMatched(rCol - 3, 0, 7 + 20, img1.rows);		//����ƥ��λ��
//	//cv::rectangle(img1, rectMatched, Scalar(255, 255, 0), 2, 8);		//��ͼ���ϻ���λ��
//	//cv::rectangle(img2, rectCut, Scalar(255, 255, 0), 2, 8);
//	////cv::Rect rectMatched((c*11) / 36, 0, (c / 9) + 20, img1.rows);		//����ƥ��λ��
//	cv::Mat imgTemp = img2(rectCut);			//����ͼ����ȡģ��
//	cv::Mat imgMatched = img1(rectMatched);			//ȡ��ͼ������������ģ��ƥ��
//
//	int width = imgMatched.cols - imgTemp.cols + 1;
//	int height = imgMatched.rows - imgTemp.rows + 1;
//	matchResult.create(height, width, CV_32FC1);
//	cv::matchTemplate(imgMatched, imgTemp, matchResult, cv::TM_CCORR_NORMED);			//ƥ��
//	//float pixel = matchResult.at<float>(0, 70);
//}
//
////4.2 //�õ����ƥ�������ƴ��
//void DoStitcher::StitImg(cv::Mat &img1, cv::Mat &img2, cv::Mat &dstImg, int angle, int BestX, int BestY)
//{
//	int c = img2.cols;	//Ϊ����չ��ͼ��Ĵ�Լ���
//	int col = (c / 2 + (c*double(angle / 180.0) / 2));
//	//����ƴ��ͼ��
//	int newRow = BestY - 40;
//	//int newRow = 0;
//	//cv::Mat dst(img1.rows , img1.cols + col - BestX, CV_8UC1);
//	//img2(cv::Rect(0, 0, col, img2.rows)).copyTo(dst(cv::Rect(0, 0, col, img2.rows)));
//	//img1(cv::Rect(BestX, 0, img1.cols - BestX, img1.rows)).copyTo(dst(cv::Rect(col, 0, img1.cols - BestX, img1.rows)));
//	cv::Mat dst(img1.rows + abs(newRow), img1.cols + col - BestX, CV_8UC1);
//
//	if (newRow <= 0) {
//		//ƴ����ͼ��
//		img2(cv::Rect(0, 0, col, img2.rows)).copyTo(dst(cv::Rect(0, 0, col, img2.rows)));
//		//ƴ�ӹ��������Ұ벿��
//		img1(cv::Rect(BestX, 0, img1.cols - BestX, img1.rows)).copyTo
//		(dst(cv::Rect(col, abs(newRow), img1.cols - BestX, img1.rows)));
//	}
//	else {
//		//ƴ����ͼ��
//		img2(cv::Rect(0, 0, col, img2.rows)).copyTo(dst(cv::Rect(0, newRow, col, img2.rows)));
//		//ƴ�ӹ��������Ұ벿��
//		img1(cv::Rect(BestX, 0, img1.cols - BestX, img1.rows)).copyTo(dst(cv::Rect(col, 0, img1.cols - BestX, img1.rows)));
//	}
//
//
//	//����ƴ�Ӻ���ں�
//	//cv::Mat leftTempImg = img2(cv::Rect(col, 0, 20, img2.rows ));
//	//cv::Mat rightTempImg = img1(Rect(BestX, BestY, 20, img1.rows ));
//	cv::Mat leftTempImg = img2(cv::Rect(col, 40, 20, img2.rows - 45));
//	cv::Mat rightTempImg = img1(Rect(BestX, BestY, 20, img1.rows - 45));
//	cv::Mat mergeImg(leftTempImg.size(), leftTempImg.type(), cv::Scalar(0));		//�ں�ͼ��
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
////��һ����β�и����ȷ�����ڣ���ƴ��ͼ�������β�иʹ����һ������
//void DoStitcher::getPeriod(cv::Mat &img01, cv::Mat &img02, int angle, int &dis, cv::Point &p)
//{
//	//���ȸ�����ת�Ƕȣ�ȷ������ƥ��λ��
//	//��ƥ����ͼ��ƥ��λ����Ϊ
//	int c = img01.cols;	//Ϊ����չ��ͼ��Ĵ�Լ���
//	//������ת�Ƕȣ�ȷ����ŵ�ƴ��λ��
//
//	int col = (c / 2 + (c*double(angle / 180.0) / 2));
//	//int col= (img01.cols / 2 + (img01.cols *double(angle / 180.0) / 2));
//	dis = img01.cols - col;			//���е�β�еľ���
//
//	//����ģ��ߴ��Լ�ƥ��λ��
//	cv::Mat tempImg = img01(Rect(col, 20, 50, img01.rows - 25));
//	cv::Mat matchImg = img02(Rect(c / 4, 0, c / 3, img02.rows));
//
//	//ģ��ƥ��
//	cv::Mat result(matchImg.rows - tempImg.rows + 1, matchImg.cols - tempImg.cols + 1, CV_32FC1);
//	cv::matchTemplate(matchImg, tempImg, result, cv::TM_CCORR_NORMED);
//	cv::normalize(result, result, 0, 1, NORM_MINMAX, -1);  //��һ����0--1��Χ
//
//	//�ҵ������Ƶ�λ��
//	double minVal, maxVal;
//	cv::Point minPt, maxPt;
//	cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);
//	p.x = maxPt.x + c / 4;
//
//	////����ͼ�ϻ���ģ������
//
//	//cv::Mat rDebugImg = img02.clone();
//	//cv::rectangle(rDebugImg, Rect(maxPt.x, maxPt.y, tempImg.cols, tempImg.rows), Scalar(255, 255, 0), 2, 8);
//	////����ͼ�ϻ���ģ������
//	//cv::Mat lDebugImg = img01.clone();
//	//cv::rectangle(lDebugImg, Rect(col, 0, tempImg.cols, tempImg.rows), Scalar(255, 255, 0), 2, 8);
//
//
//}