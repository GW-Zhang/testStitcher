#include "shape_match.h"
#include <math.h>

#define SM_EPS 0.00001f
#define SM_PI	3.1415926535897932384626433832795f
#define SM_MODEL_SUFFUX string(".yaml")

const float AngleRegionTable[16][2] = {

	0.f		, 22.5f	,
	22.5f	, 45.f	,
	45.f	, 67.5f	,
	67.5f	, 90.f	,
	90.f	, 112.5f,
	112.5f	, 135.f	,
	135.f	, 157.5f,
	157.5f	, 180.f,
	180.f	, 202.5f,
	202.5f	, 225.f,
	225.f	, 247.5f,
	247.5f	, 270.f,
	270.f	, 292.5f,
	292.5f	, 315.f,
	315.f	, 337.5f,
	337.5f	, 360.f
};

namespace cv_dnn_nms {

	template <typename T>
	static inline bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2) {

		return pair1.first > pair2.first;
	}

	inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
		std::vector<std::pair<float, int> >& score_index_vec) {

		for (size_t i = 0; i < scores.size(); ++i)
		{
			if (scores[i] > threshold)
			{
				//score_index_vec.push_back(std::make_pair(scores[i], i));
				std::pair<float, int> psi;
				psi.first = scores[i];
				psi.second = (int)i;
				score_index_vec.push_back(psi);
			}
		}
		std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
			SortScorePairDescend<int>);
		if (top_k > 0 && top_k < (int)score_index_vec.size())
		{
			score_index_vec.resize(top_k);
		}
	}

	template <typename BoxType>
	inline void NMSFast_(const std::vector<BoxType>& bboxes,
		const std::vector<float>& scores, const float score_threshold,
		const float nms_threshold, const float eta, const int top_k,
		std::vector<int>& indices, float(*computeOverlap)(const BoxType&, const BoxType&)) {

		CV_Assert(bboxes.size() == scores.size());
		std::vector<std::pair<float, int> > score_index_vec;

		//将其进行排序，选出得分最高的前top_k个
		GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

		float adaptive_threshold = nms_threshold;
		indices.clear();

		// 对于本次的这个得分最大值序号idx，只要其他候选框的重叠度都小于adaptive_threshold
// 那就证明不是同一个目标，那就保留这个局部最大值，直到把得分top_k的框都遍历一遍
		for (size_t i = 0; i < score_index_vec.size(); ++i) {
			const int idx = score_index_vec[i].second;
			bool keep = true;
			for (int k = 0; k < (int)indices.size() && keep; ++k) {
				const int kept_idx = indices[k];
				float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);		//计算出候选框的重叠度
				keep = overlap <= adaptive_threshold;		//判断候选框重叠度是否小于阈值（若小于阈值，则认为其是最大的）
			}

			// 如果把所有的框都遍历一遍，没有跟idx有重叠超过阈值的，说明它是局部极大值，保留。
			if (keep)
				indices.push_back(idx);
			if (keep && eta < 1 && adaptive_threshold > 0.5) {
				//自适应阈值的衰减系数
				adaptive_threshold *= eta;
			}
		}
	}

	template<typename _Tp> static inline
		double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
		_Tp Aa = a.area();
		_Tp Ab = b.area();
		//返回的是计算机体系结构所能判断的两个同类型的数据是否相等的极限
		if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
			// jaccard_index = 1 -> distance = 0
			return 0.0;
		}

		//a&b表示a与b共有的区域，交集
		double Aab = (a & b).area();
		// distance = 1 - jaccard_index
		return 1.0 - Aab / (Aa + Ab - Aab);
	}

	template <typename T>
	static inline float rectOverlap(const T& a, const T& b) {

		return 1.f - static_cast<float>(jaccardDistance__(a, b));
	}

	void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
		const float score_threshold, const float nms_threshold,
		std::vector<int>& indices, const float eta = 1, const int top_k = 0) {

		NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
	}

} // end namespace cv_dnn_nms

namespace shapematch {

	ShapeMatching::ShapeMatching(string model_root, string class_name) {

		assert(!model_root.empty() && "model_root should not empty.");
		assert(!class_name.empty() && "class_name should not empty.");
		if (model_root[model_root.length() - 1] != '/') {

			model_root.push_back('/');
		}
		model_root_ = model_root;
		class_name_ = class_name;

		/// 创造 180*180 表格
		for (int i = 0; i < 180; i++) {

			for (int j = 0; j < 180; j++) {

				float rad = (i - j) * SM_PI / 180.f;
				score_table_[i][j] = fabs(cosf(rad));
			}
		}

		//建立8*8 表格
		ATTR_ALIGN(8) unsigned char score_table_8d[8][8];
		for (int i = 0; i < 8; i++) {

			for (int j = 0; j < 8; j++) {

				float rad = (i - j) * (180.f / 8.f) * SM_PI / 180.f;
				score_table_8d[i][j] = (unsigned char)(fabs(cosf(rad))*100.f);
			}
		}

		//建立8*256表格
		for (int i = 0; i < 8; i++) {

			for (int j = 0; j < 256; j++) {

				unsigned char max_score = 0;
				for (int shift_time = 0; shift_time < 8; shift_time++) {

					unsigned char flg = (j >> shift_time) & 0b00000001;
					if (flg) {

						if (score_table_8d[i][shift_time] > max_score) {

							max_score = score_table_8d[i][shift_time];//计算出最大的量化角度
						}
					}
				}
				score_table_8map_[i][j] = max_score;
			}
		}
	}

	ShapeMatching::~ShapeMatching() {

	}

	//计算角点响应图，并进行处理
	cv::Mat ShapeMatching::calcuCorner(cv::Mat &image)
	{
		// 计算图像的角点响应值
		cv::Mat corner_response;
		cv::cornerHarris(image, corner_response, 10, 3, 0.04);

		cv::Mat cornersNormalized;
		cv::normalize(corner_response, cornersNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
	
		cv::Mat unit_cornerNormalized;
		cornersNormalized.convertTo(unit_cornerNormalized, CV_8UC1);

		return unit_cornerNormalized;
	}

	//1.制作模板
	void ShapeMatching::MakingTemplates(Mat &modelRGB, AngleRange angle_range, ScaleRange scale_range,
		int num_features, float weak_thresh, float strong_thresh, Mat mask) {

		////转灰度或者寻找最佳通道或寻找HSV最佳通道
		cv::Mat model, channelModel;
		cv::cvtColor(modelRGB, model, COLOR_BGR2GRAY);

		cv::blur(model, model, cv::Size(3, 3));


		model_ = model;
		ClearModel();

		weak_thresh_ = weak_thresh;
		strong_thresh_ = strong_thresh;

		girdLength(model, 2);

		//计算角点响应强度
		cv::Mat cornerResponse = calcuCorner(model);

		//对model和mask进行填充操作
		PaddingModelAndMask(cornerResponse,model, mask, scale_range.end);

		//初始化角度、尺度范围
		angle_range_ = angle_range;
		scale_range_ = scale_range;

		//生成所有的模板信息
		vector<ShapeInfo> shape_infos = ProduceShapeInfos(angle_range, scale_range);//不同角度尺度模板信息（存储的是角度，尺度）
		vector<Mat> l0_mdls; l0_mdls.clear();
		vector<Mat> l0_msks; l0_msks.clear();
		vector<Mat>l0_cors; l0_cors.clear();

		//生成所有模板的底层金字塔图像
		for (int s = 0; s < shape_infos.size(); s++) {

			l0_mdls.push_back(MdlOf(model, shape_infos[s]));
			l0_msks.push_back(MskOf(mask, shape_infos[s]));
			l0_cors.push_back(MdlOf(cornerResponse, shape_infos[s]));
		}

		//对所有层的金字塔图像进行特征提取
		for (int p = 0; p <= PyramidLevel_7; p++) {
			//某层金字塔的所有角度、尺度图像
			for (int s = 0; s < shape_infos.size(); s++) {

				Mat mdl_pyrd = l0_mdls[s];
				Mat msk_pyrd = l0_msks[s];
				Mat cor_pyrd = l0_cors[s];
				if (p > 0) {

					Size sz = Size(l0_mdls[s].cols >> 1, l0_mdls[s].rows >> 1);
					pyrDown(l0_mdls[s], mdl_pyrd, sz);
					pyrDown(l0_msks[s], msk_pyrd, sz);
					pyrDown(l0_cors[s], cor_pyrd, sz);
				}
				//对图像进行腐蚀操作，因为有效信息可能在边缘，利用sobel可能提取不到
				erode(msk_pyrd, msk_pyrd, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
				l0_mdls[s] = mdl_pyrd;
				l0_msks[s] = msk_pyrd;
				l0_cors[s] = cor_pyrd;

				//计算某层金字塔需要的特征数量
				int features_pyrd = (int)((num_features >> p) * shape_infos[s].scale);

				Mat mag8, angle8, quantized_angle8;
				//量化梯度角度（8个方向）
				QuantifyEdge(mdl_pyrd, angle8, quantized_angle8, mag8, weak_thresh, false);
				//提取模板信息（8个方向）
				Template templ = ExtractTemplate(cor_pyrd,angle8, quantized_angle8, mag8,
					shape_infos[s], PyramidLevel(p),
					weak_thresh, strong_thresh,
					features_pyrd, msk_pyrd);
				templ_all_[p].push_back(templ);


				//显示训练结果
				if (p == 0)
				{
					cv::Mat tempImg = mdl_pyrd.clone();
					cv::cvtColor(tempImg, tempImg, COLOR_GRAY2BGR);
					for (int i = 0; i < templ.features.size(); i++)
					{
						cv::circle(tempImg,
							cv::Point(templ.features[i].x+min_x_, templ.features[i].y+min_y_)
							, 2, cv::Scalar(255, 0, 255), -1);
					}
					imshow("", tempImg);
					waitKey(10);

				}

				////画出轮廓点
				//for (int i = 0; i < (int)templ.features.size(); i++) {

				//	auto feature = templ.features[i];
				//	//画出轮廓点;
				//	line(model,
				//		Point(feature.x+70, feature.y+78),
				//		Point(feature.x + 70, feature.y + 78),
				//		cv::Scalar(255, 255,255), 3);
				//}
				//for (int r = 0; r < modelRGB.rows; r += 8)
				//{
				//	cv::line(modelRGB, Point(0, r), Point(modelRGB.cols, r), cv::Scalar(0, 0, 255), 1);
				//}
				//for (int c = 0; c < modelRGB.cols; c += 8)
				//{
				//	cv::line(modelRGB, Point(c, 0), Point(c, modelRGB.rows), cv::Scalar(0, 0, 255), 1);
				//}
				//cv::imshow("templ point", model);
				//cv::waitKey(0);

				Mat mag180, angle180, quantized_angle180;
				QuantifyEdge(mdl_pyrd, angle180, quantized_angle180, mag180, weak_thresh, true);
				templ = ExtractTemplate(model,angle180, quantized_angle180, mag180,
					shape_infos[s], PyramidLevel(p),
					weak_thresh, strong_thresh,
					features_pyrd, msk_pyrd);
				templ_all_[p + 8].push_back(templ);

			}
			cout << "train pyramid level " << p << " complete." << endl;
		}
		//对模板信息进行保存
		SaveModel();
	}

	vector<Match> ShapeMatching::Matching(Mat sourceRGB, float score_thresh, float overlap,
		float mag_thresh, float greediness, PyramidLevel pyrd_level, int T, int top_k,
		MatchingStrategy strategy, const Mat mask) {

		//cv::Mat source= sourceHSVChannel(sourceRGB, modelHSV_.channel);
		//blur(source, source, Size(9, 9));
		cv::Mat source;
		cv::cvtColor(sourceRGB, source, COLOR_BGR2GRAY);
		cv::blur(source, source, cv::Size(3, 3));

		//初始化匹配参数
		InitMatchParameter(score_thresh, overlap, mag_thresh, greediness, T, top_k, strategy);

		//获取搜索图像进行所有的有效金字塔层图像
		GetAllPyramidLevelValidSource(source, pyrd_level);

		vector<Match> matches;
		// 从最高层金字塔开始匹配量化为8方向的图像相似度矩阵，（已经排好序）
		matches = MatchingPyrd8(sources_[pyrd_level], pyrd_level, false, region8_idxes_);
		//挑选出前k个得分最高的匹配对
		matches = GetTopKMatches(matches);

		//再确认匹配对
		matches = ReconfirmMatches(matches, pyrd_level);
		matches = GetTopKMatches(matches);

		//最终确认匹配对（量化180方向）
		matches = MatchingFinal(matches, pyrd_level);
		matches = GetTopKMatches(matches);

		return matches;
	}

	////画出匹配后图像
	//void ShapeMatching::DrawMatches(Mat &image, vector<Match> matches, Scalar color) {

	//	
	//	for (int i = 0; i < matches.size(); i++) {

	//		auto match = matches[i];
	//		auto templ = templ_all_[8][match.template_id];
	//		int w = match.x + templ.w;
	//		int h = match.y + templ.h;
	//		for (int i = 0; i < (int)templ.features.size(); i++) {

	//			auto feature = templ.features[i];
	//			//画出轮廓点;
	//			line(image,
	//				Point(match.x + feature.x, match.y + feature.y),
	//				Point(match.x + feature.x, match.y + feature.y),
	//				color, 4);
	//		}
	//		cv::rectangle(image, { match.x, match.y }, { w, h }, color, 1);//画出矩形
	//		char info[128];//标出匹配度，旋转角度，尺度，存储在info中
	//		sprintf(info,
	//			"%.2f%% [%.2f, %.2f]",
	//			match.similarity * 100,
	//			templ.shape_info.angle,
	//			templ.shape_info.scale);
	//		//在图像上进行文字绘制
	//		cv::putText(image,
	//			info,
	//			Point(match.x, match.y), FONT_HERSHEY_PLAIN, 1.f, color, 1);
	//	}
	//}



		//画出匹配后图像
	void ShapeMatching::DrawMatches(Mat &image, vector<Match> matches, Scalar color) {


		for (int i = 0; i < matches.size(); i++) {

			auto match = matches[i];
			auto templ = templ_all_[8][match.template_id];
			match.x = match.x - templ.w / 2;
			match.y = match.y - templ.h / 2;
			int w = match.x + templ.w;
			int h = match.y + templ.h;
			//int w = match.x + model_.cols;
			//int h = match.y + model_.rows;
			int centerX = match.x + templ.w / 2;
			int centerY = match.y + templ.h / 2;

			cv::Point leftUp, rightUp, leftDown, rightDown;
			//leftUp.x = (match.x-centerX)*double(cos(templ.shape_info.angle))+(match.y-centerY)*double(sin(templ.shape_info.angle))+centerX;
			//leftUp.y= -(match.x - centerX)*double(sin(templ.shape_info.angle)) + (match.y - centerY)*double(cos(templ.shape_info.angle)) + centerY;

			//rightUp.x = (w - centerX)*double(cos(templ.shape_info.angle)) + (match.y - centerY)*double(sin(templ.shape_info.angle)) + centerX;
			//rightUp.y = -(w - centerX)*double(sin(templ.shape_info.angle)) + (match.y - centerY)*double(cos(templ.shape_info.angle)) + centerY;

			//leftDown.x = (match.x - centerX)*double(cos(templ.shape_info.angle)) + (h - centerY)*double(sin(templ.shape_info.angle)) + centerX;
			//leftDown.y = -(match.x - centerX)*double(sin(templ.shape_info.angle)) + (h - centerY)*double(cos(templ.shape_info.angle)) + centerY;

			//rightDown.x = (w - centerX)*double(cos(templ.shape_info.angle)) + (h - centerY)*double(sin(templ.shape_info.angle)) + centerX;
			//rightDown.y = -(w - centerX)*double(sin(templ.shape_info.angle)) + (h - centerY)*double(cos(templ.shape_info.angle)) + centerY;

			//leftUp.x = match.x*cos(templ.shape_info.angle) + match.y*sin(templ.shape_info.angle);
			//leftUp.y=-match.x*sin(templ)

			cv::RotatedRect box(cv::Point(centerX - templ.relaCenter.x, centerY - templ.relaCenter.y),
				cv::Size(model_.cols, model_.rows), -templ.shape_info.angle);
			cv::Point2f vertex[4];
			box.points(vertex);
			for (int i = 0; i < 4; i++)
				line(image, vertex[i], vertex[(i + 1) % 4], Scalar(0, 255, 0), 4);

			arrowedLine(image, Point2f((vertex[1].x + vertex[2].x) / 2, (vertex[1].y + vertex[2].y) / 2),
				cv::Point(centerX - templ.relaCenter.x, centerY - templ.relaCenter.y), Scalar(0, 255, 0), 4, 8, 0, 0.2);
			//line(image, leftUp, leftDown, color, 3);
			//line(image, leftUp, rightUp, color, 3);
			//line(image, rightUp, rightDown, color, 3);
			//line(image, leftDown, rightDown, color, 3);

			for (int i = 0; i < (int)templ.features.size(); i++) {

				auto feature = templ.features[i];
				//画出轮廓点;
				line(image,
					Point(match.x + feature.x, match.y + feature.y),
					Point(match.x + feature.x, match.y + feature.y),
					color, 4);


			}
			//cv::rectangle(image, { match.x, match.y }, { w, h }, color, 1);//画出矩形
			//char info[128];//标出匹配度，旋转角度，尺度，存储在info中
			//sprintf(info,
			//	"%.2f%% [%.2f, %.2f]",
			//	match.similarity * 100,
			//	templ.shape_info.angle,
			//	templ.shape_info.scale);
			////在图像上进行文字绘制
			//cv::putText(image,
			//	info,
			//	Point(match.x, match.y), FONT_HERSHEY_PLAIN, 1.f, color, 1);

		}
	}

	//进行扩充，方便模板图像旋转放缩
	void ShapeMatching::PaddingModelAndMask(cv::Mat &corner,Mat &model, Mat &mask, float max_scale) {

		model_ = model;
		CV_Assert(!model.empty() && "model is empty.");
		if (mask.empty())
			mask = Mat(model.size(), CV_8UC1, { 255 });
		else
			CV_Assert(model.size() == mask.size());
		int min_side_length = std::min(model.rows, model.cols);
		//求解对角线长度
		int diagonal_line_length =
			(int)ceil(std::sqrt(model.rows*model.rows + model.cols*model.cols)*max_scale);//ceil用于对float,double,longdouble向上取整，且最后结果为double类型，不是int类型
		int padding = ((diagonal_line_length - min_side_length) >> 1) + 16;
		padding_ = padding;
		int double_padding = (padding << 1);
		Mat model_padded = Mat(model.rows + double_padding, model.cols + double_padding, model.type(), Scalar::all(0));
		model.copyTo(model_padded(Rect(padding, padding, model.cols, model.rows)));
		Mat mask_padded = Mat(mask.rows + double_padding, mask.cols + double_padding, mask.type(), Scalar::all(0));
		mask.copyTo(mask_padded(Rect(padding, padding, mask.cols, mask.rows)));
		Mat corner_padded = Mat(corner.rows + double_padding, corner.cols + double_padding, mask.type(), Scalar::all(0));
		corner.copyTo(corner_padded(Rect(padding, padding, corner.cols, corner.rows)));
		model = model_padded;
		mask = mask_padded;
		corner = corner_padded;
	}

	//生成各种模板信息
	vector<ShapeInfo> ShapeMatching::ProduceShapeInfos(AngleRange angle_range, ScaleRange scale_range) {

		assert(scale_range.begin > SM_EPS && scale_range.end > SM_EPS);
		assert(angle_range.end >= angle_range.begin);
		assert(scale_range.end >= scale_range.begin);
		assert(angle_range.step > SM_EPS);
		assert(scale_range.step > SM_EPS);
		vector<ShapeInfo> shape_infos;
		shape_infos.clear();
		for (float scale = scale_range.begin; scale <= scale_range.end + SM_EPS; scale += scale_range.step) {

			for (float angle = angle_range.begin; angle <= angle_range.end + SM_EPS; angle += angle_range.step) {

				ShapeInfo info;
				info.angle = angle;
				info.scale = scale;
				shape_infos.push_back(info);
			}
		}
		return shape_infos;
	}

	//对模板图像进行转换生成不同的模板
	Mat ShapeMatching::Transform(Mat src, float angle, float scale) {

		Mat dst;
		Point center(src.cols / 2, src.rows / 2);
		Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
		warpAffine(src, dst, rot_mat, src.size());
		return dst;
	}

	Mat ShapeMatching::MdlOf(Mat model, ShapeInfo info) {

		return Transform(model, info.angle, info.scale);
	}

	Mat ShapeMatching::MskOf(Mat mask, ShapeInfo info) {

		return (Transform(mask, info.angle, info.scale) > 0);
	}


	void ShapeMatching::DrawTemplate(Mat &image, Template templ, Scalar color) {

		for (int i = 0; i < templ.features.size(); i++) {

			auto feature = templ.features[i];
			line(image,
				Point(templ.x + feature.x, templ.y + feature.y),
				Point(templ.x + feature.x, templ.y + feature.y),
				color, 1);
		}
	}

	//量化梯度角度
	void ShapeMatching::QuantifyEdge(Mat image, Mat &angle, Mat &quantized_angle, Mat &mag, float mag_thresh, bool calc_180) {

		Mat dx, dy;
		float mask_x[3][3] = { { -1,0,1 },{ -2,0,2 },{ -1,0,1 } };
		float mask_y[3][3] = { { 1,2,1 },{ 0,0,0 },{ -1,-2,-1 } };
		Mat kernel_x = Mat(3, 3, CV_32F, mask_x);
		Mat kernel_y = Mat(3, 3, CV_32F, mask_y);
		filter2D(image, dx, CV_32F, kernel_x);
		filter2D(image, dy, CV_32F, kernel_y);
		//dx = abs(dx);
		//dy = abs(dy);
		mag = dx.mul(dx) + dy.mul(dy);	//幅值计算
		phase(dx, dy, angle, true);		//幅角计算

		if (calc_180)	//量化180
			Quantify180(angle, quantized_angle, mag, mag_thresh);
		else //量化8方向
			Quantify8(angle, quantized_angle, mag, mag_thresh);
	}

	//将梯度角度量化为8个方向
	void ShapeMatching::Quantify8(Mat angle, Mat &quantized_angle, Mat mag, float mag_thresh) {

		Mat_<unsigned char> quantized_unfiltered;
		angle.convertTo(quantized_unfiltered, CV_8U, 16.0f / 360.0f);		//将量化角度归一化在0-16之间
		for (int r = 0; r < angle.rows; ++r)
		{
			unsigned char *quant_ptr = quantized_unfiltered.ptr<unsigned char>(r);
			for (int c = 0; c < angle.cols; ++c)
			{
				quant_ptr[c] &= 7;		//按位与操作，将梯度方向量化到8个方向上，将大于8的方向反射到8以内
			}
		}
		//quantized_unfiltered.copyTo(quantized_angle);
		quantized_angle = Mat::zeros(angle.size(), CV_8U);
		for (int r = 0; r < quantized_angle.rows; ++r) {

			quantized_angle.ptr<unsigned char>(r)[0] = 255;
			quantized_angle.ptr<unsigned char>(r)[quantized_angle.cols - 1] = 255;
		}
		for (int c = 0; c < quantized_angle.cols; ++c) {

			quantized_angle.ptr<unsigned char>(0)[c] = 255;
			quantized_angle.ptr<unsigned char>(quantized_angle.rows - 1)[c] = 255;
		}

		for (int r = 1; r < angle.rows - 1; ++r)
		{
			float *mag_ptr = mag.ptr<float>(r);
			for (int c = 1; c < angle.cols - 1; ++c)
			{
				//需要满足幅值大于最小阈值，才进行接下来的比较
				if (mag_ptr[c] >= (mag_thresh * mag_thresh))
				{
					//计算像素点周围3*3区域的方向直方图累计，就是看看哪个方向的趋势最大
					int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

					//统计出角度落在每个bin中的数量
					unsigned char *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					patch3x3_row += quantized_unfiltered.step1();
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					patch3x3_row += quantized_unfiltered.step1();
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					//寻找最大方向趋势的下标
					int max_votes = 0;
					int index = -1;
					for (int i = 0; i < 8; ++i)
					{
						if (max_votes < histogram[i])
						{
							index = i;
							max_votes = histogram[i];
						}
					}

					//如果该方向的趋势足够大（>5），则记录该像素的方向，这里进行了放大，把1向左移动几位，就是放大2的多少倍
					static const int NEIGHBOR_THRESHOLD = 5;
					if (max_votes >= NEIGHBOR_THRESHOLD)
						quantized_angle.at<unsigned char>(r, c) = index;
					else
						quantized_angle.at<unsigned char>(r, c) = 255;
				}
				else
				{
					quantized_angle.at<unsigned char>(r, c) = 255;
				}
			}
		}
	}

	//将梯度角度量化到180°区间方向上
	void ShapeMatching::Quantify180(Mat angle, Mat &quantized_angle, Mat mag, float mag_thresh) {

		quantized_angle = Mat::zeros(angle.size(), CV_8U);
#pragma omp parallel for
		for (int r = 0; r < angle.rows; ++r)
		{
			unsigned char *quantized_angle_ptr = quantized_angle.ptr<unsigned char>(r);
			float *angle_ptr = angle.ptr<float>(r);
			float *mag_ptr = mag.ptr<float>(r);
			for (int c = 0; c < angle.cols; ++c)
			{
				//如果幅值满足大于最小阈值，则进行接下来的处理
				if (mag_ptr[c] >= (mag_thresh * mag_thresh))
					//将大于180°的角度反射回180°以内
					quantized_angle_ptr[c] = (int)round(angle_ptr[c]) % 180;
				else
					quantized_angle_ptr[c] = 255;
			}
		}
	}

	//双阈值设置提取边缘，canny算子
	Template ShapeMatching::ExtractTemplate(Mat cor_pyrd,Mat angle, Mat quantized_angle, Mat mag, ShapeInfo shape_info,
		PyramidLevel pl, float weak_thresh, float strong_thresh, int num_features, Mat mask) {

		Mat local_angle = Mat(angle.size(), angle.type());
		for (int r = 0; r < angle.rows; ++r) {

			float *angle_ptr = angle.ptr<float>(r);
			float *local_angle_ptr = local_angle.ptr<float>(r);
			for (int c = 0; c < angle.cols; ++c) {

				float dir = angle_ptr[c];
				if ((dir > 0. && dir < 22.5) || (dir > 157.5 && dir < 202.5) || (dir > 337.5 && dir < 360.))
					local_angle_ptr[c] = 0.f;
				else if ((dir > 22.5 && dir < 67.5) || (dir > 202.5 && dir < 247.5))
					local_angle_ptr[c] = 45.f;
				else if ((dir > 67.5 && dir < 112.5) || (dir > 247.5 && dir < 292.5))
					local_angle_ptr[c] = 90.f;
				else if ((dir > 112.5 && dir < 157.5) || (dir > 292.5 && dir < 337.5))
					local_angle_ptr[c] = 135.f;
				else
					local_angle_ptr[c] = 0.f;
			}
		}

		vector<Candidate> candidates;
		candidates.clear();
		bool no_mask = mask.empty();
		float weak_sq = weak_thresh * weak_thresh;
		float strong_sq = strong_thresh * strong_thresh;
		float pre_grad, lst_grad;
		for (int r = 1; r < mag.rows - 1; ++r)
		{
			const unsigned char *mask_ptr = no_mask ? NULL : mask.ptr<unsigned char>(r);
			const float* pre_ptr = mag.ptr<float>(r - 1);
			const float* cur_ptr = mag.ptr<float>(r);
			const float* lst_ptr = mag.ptr<float>(r + 1);
			float *local_angle_ptr = local_angle.ptr<float>(r);

			for (int c = 1; c < mag.cols - 1; ++c)
			{
				if (no_mask || mask_ptr[c])
				{
					switch ((int)local_angle_ptr[c]) {

					case 0:
						pre_grad = cur_ptr[c - 1];
						lst_grad = cur_ptr[c + 1];
						break;
					case 45:
						pre_grad = pre_ptr[c + 1];
						lst_grad = lst_ptr[c - 1];
						break;
					case 90:
						pre_grad = pre_ptr[c];
						lst_grad = lst_ptr[c];
						break;
					case 135:
						pre_grad = pre_ptr[c - 1];
						lst_grad = lst_ptr[c + 1];
						break;
					}
					if ((cur_ptr[c] > pre_grad) && (cur_ptr[c] > lst_grad)) {

						float score = cur_ptr[c];
						bool validity = false;
						if (score >= weak_sq) {

							if (score >= strong_sq) {

								validity = true;
							}
							else {

								if (((pre_ptr[c - 1]) >= strong_sq) ||
									((pre_ptr[c]) >= strong_sq) ||
									((pre_ptr[c + 1]) >= strong_sq) ||
									((cur_ptr[c - 1]) >= strong_sq) ||
									((cur_ptr[c + 1]) >= strong_sq) ||
									((lst_ptr[c - 1]) >= strong_sq) ||
									((lst_ptr[c]) >= strong_sq) ||
									((lst_ptr[c + 1]) >= strong_sq))
								{
									validity = true;
								}
							}
						}
						if (validity == true &&
							quantized_angle.at<unsigned char>(r, c) != 255) {

							Candidate cd;
							cd.score = score;
							cd.feature.x = c;
							cd.feature.y = r;
							cd.feature.lbl = quantized_angle.at<unsigned char>(r, c);
							candidates.push_back(cd);
						}
					}

				}
			}
		}

		Template templ;
		templ.shape_info.angle = shape_info.angle;
		templ.shape_info.scale = shape_info.scale;
		templ.pyramid_level = pl;
		templ.is_valid = 0;
		templ.InitEdgeNum = candidates.size();
		templ.features.clear();

		if (candidates.size() >= num_features && num_features > 0) {

			std::stable_sort(candidates.begin(), candidates.end());
			float distance = static_cast<float>(candidates.size() / num_features + 1);
			templ = SelectScatteredFeatures(candidates, num_features, distance);

		}
		else {
			//网格化筛选
			std::vector<Candidate>newCandidates = gridSelectFeaturePoints(cor_pyrd,angle, quantized_angle, mag, candidates);
			std::vector<cv::Point2f>vPt;
			for (int c = 0; c < newCandidates.size(); c++) {
				vPt.push_back(cv::Point2f(newCandidates[c].feature.x, newCandidates[c].feature.y));
				templ.features.push_back(newCandidates[c].feature);
			}
			KCluster kc;
			//kc.doCluster(angle,mag,angle, vPt, 4);
		}
		//cv::Mat edgeImg = cv::Mat::zeros(angle.size(),CV_8UC1);
		//for (const auto& i : templ.features)
		//{
		//	circle(edgeImg, cv::Point(i.x,i.y), 1, Scalar(100), FILLED, LINE_AA);
		//}


		if (templ.features.size() > 0) {

			templ.is_valid = 1;
			CropTemplate(templ, angle);
		}

		return templ;
	}

	//网格化长度的计算
	void ShapeMatching::girdLength(cv::Mat model, int pl)
	{
		//计算出网格的size
		int min_side = min(model.cols / pow(2, pl), model.rows / pow(2, pl));
		int grid_length = min_side / 8;	//自适应最小网格长度
		grid_length_ = grid_length;
	}

	//网格化筛选特征点
	std::vector<Candidate> ShapeMatching::gridSelectFeaturePoints(cv::Mat cornerResponse,
		cv::Mat angle,cv::Mat quantized_angle, cv::Mat mag, std::vector<Candidate> candidates)
	{

		std::vector<Candidate> newCandidates;
		//计算出网格的size
		int grid_length = grid_length_;	//自适应最小网格长度

		//进行网格化筛选
		cv::Mat candidateImg = cv::Mat::zeros(angle.size(), CV_8UC1);
		for (int i = 0; i < candidates.size(); i++)
		{
			candidateImg.at<uchar>(candidates[i].feature.y, candidates[i].feature.x) = 255;
		}

		for (int r = 0; r < angle.rows - grid_length; r += grid_length)
		{
			for (int c = 0; c < angle.cols - grid_length; c += grid_length)
			{
				cv::Point pt = cv::Point(0, 0);
				//double iterNum = dst.at<uchar>(r + grid_length, c + grid_length) + dst.at<uchar>(r, c) -
				   //		dst.at<uchar>(r + grid_length, c) - dst.at<uchar>(r , c + grid_length);
					   //if (iterNum!=0)
					   //{

				float max = 0.0;
				int num = 0;
				for (int grid_r = r; grid_r < r + grid_length; grid_r++)
				{	
					for (int grid_c = c; grid_c < c + grid_length; grid_c++)
					{
						if (candidateImg.at<uchar>(grid_r, grid_c) == 255)
						{
							if (cornerResponse.at<uchar>(grid_r, grid_c) > 250)
							{
								pt.x = grid_c;
								pt.y = grid_r;
								num++;
								Candidate cd;
								cd.score = mag.at<float>(pt.y, pt.x);
								cd.feature.x = pt.x;
								cd.feature.y = pt.y;
								cd.feature.lbl = quantized_angle.at<unsigned char>(pt.y, pt.x);
								newCandidates.push_back(cd);
							}
							else {
								continue;
							}

						}
					}
				}
				if (num == 0)
				{
					for (int grid_r = r; grid_r < r + grid_length; grid_r++)
					{
						for (int grid_c = c; grid_c < c + grid_length; grid_c++) {
							if (candidateImg.at<uchar>(grid_r, grid_c) == 255) {
								float result = (0.8)*mag.at<float>(grid_r, grid_c) + (0.2)*angle.at<float>(grid_r, grid_c);
								if (result > max)
								{
									max = result;
									pt.x = grid_c;
									pt.y = grid_r;
								}
							}
						}
					}
					if (mag.at<float>(pt.y, pt.x) != 0)
					{
						Candidate cd;
						cd.score = mag.at<float>(pt.y, pt.x);
						cd.feature.x = pt.x;
						cd.feature.y = pt.y;
						cd.feature.lbl = quantized_angle.at<unsigned char>(pt.y, pt.x);
						newCandidates.push_back(cd);
					}
				}	//if(num==0)
			}
		}

		return newCandidates;
	}

	//挑选候选者
	Template ShapeMatching::SelectScatteredFeatures(vector<Candidate> candidates, int num_features, float distance) {

		Template templ;
		templ.features.clear();
		float distance_sq = distance * distance;
		int i = 0;
		while (templ.features.size() < num_features) {

			Candidate c = candidates[i];
			//如果距离任何先前选择的特征足够远，则添加
			bool keep = true;
			for (int j = 0; (j < (int)templ.features.size()) && keep; ++j)
			{
				Feature f = templ.features[j];
				keep = ((c.feature.x - f.x) * (c.feature.x - f.x) + (c.feature.y - f.y) * (c.feature.y - f.y) >= distance_sq);
			}
			if (keep)
				templ.features.push_back(c.feature);

			if (++i == (int)candidates.size())
			{
				i = 0;
				distance -= 1.0f;
				distance_sq = distance * distance;

			}
		}
		return templ;
	}

	//裁剪模板
	Rect ShapeMatching::CropTemplate(Template &templ, cv::Mat angle) {

		int min_x = std::numeric_limits<int>::max();
		int min_y = std::numeric_limits<int>::max();
		int max_x = std::numeric_limits<int>::min();
		int max_y = std::numeric_limits<int>::min();

		//找到最佳特征点 
		for (int i = 0; i < (int)templ.features.size(); ++i)
		{
			int x = templ.features[i].x;
			int y = templ.features[i].y;
			min_x = std::min(min_x, x);
			min_y = std::min(min_y, y);
			max_x = std::max(max_x, x);
			max_y = std::max(max_y, y);
		}


		if (min_x % 2 == 1)
			--min_x;
		if (min_y % 2 == 1)
			--min_y;

		// 特征点相对位置坐标变换
		cv::Point initCenter = cv::Point(angle.cols / 2, angle.rows / 2);

		templ.w = (max_x - min_x);
		templ.h = (max_y - min_y);
		templ.x = min_x;
		templ.y = min_y;
		cv::Point endCenter = cv::Point(templ.w / 2 + templ.x, templ.h / 2 + templ.y);
		cv::Point relative_center = cv::Point(endCenter.x - initCenter.x, endCenter.y - initCenter.y);
		templ.relaCenter = relative_center;

		for (int i = 0; i < (int)templ.features.size(); ++i)
		{
			templ.features[i].x -= templ.x;
			templ.features[i].y -= templ.y;
		}
		min_x_ = min_x;
		min_y_ = min_y;
		return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	}

	void ShapeMatching::LoadRegion8Idxes() {

		int keys[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
		region8_idxes_.clear();
		int angle_region = (int)((angle_range_.end - angle_range_.begin) / angle_range_.step) + 1;
		int scale_region = (int)((scale_range_.end - scale_range_.begin) / scale_range_.step) + 1;
		for (int ar = 0; ar < angle_region; ar++) {

			float cur_agl = templ_all_[PyramidLevel_0][ar].shape_info.angle;
			if (cur_agl < 0.f) cur_agl += 360.f;
			int idx = 0;
			for (int i = 0; i < 16; i++) {

				if (cur_agl >= AngleRegionTable[i][0] &&
					cur_agl < AngleRegionTable[i][1]) {

					idx = i;
					break;
				}
			}
			if (keys[idx] == 0) {

				for (int sr = 0; sr < scale_region; sr++) {

					region8_idxes_.push_back(ar + sr * angle_region);
				}
			}
			keys[idx] = 1;
		}
	}

	void ShapeMatching::SaveModel() {

		int total_templ = 0;
		for (int i = 0; i < PyramidLevel_TabooUse; i++) {

			total_templ += (int)templ_all_[i].size();
		}
		assert((total_templ / PyramidLevel_TabooUse) == templ_all_[0].size());
		int match_range_size = (int)templ_all_[0].size();
		string model_name = model_root_ + class_name_ + SM_MODEL_SUFFUX;
		FileStorage fs(model_name, FileStorage::WRITE);
		fs << "class_name" << class_name_;
		fs << "total_pyramid_levels" << PyramidLevel_7;
		fs << "angle_range_bgin" << angle_range_.begin;
		fs << "angle_range_end" << angle_range_.end;
		fs << "angle_range_step" << angle_range_.step;
		fs << "scale_range_bgin" << scale_range_.begin;
		fs << "scale_range_end" << scale_range_.end;
		fs << "scale_range_step" << scale_range_.step;
		fs << "templates"
			<< "[";
		{
			for (int i = 0; i < match_range_size; i++) {

				fs << "{";
				fs << "template_id" << int(i);
				fs << "template_pyrds"
					<< "[";
				{
					for (int j = 0; j < PyramidLevel_TabooUse; j++) {

						auto templ = templ_all_[j][i];
						fs << "{";
						fs << "id" << int(i);
						fs << "pyramid_level" << templ.pyramid_level;
						fs << "is_valid" << templ.is_valid;
						fs << "x" << templ.x;
						fs << "y" << templ.y;
						fs << "w" << templ.w;
						fs << "h" << templ.h;
						fs << "relaCenter_X" << templ.relaCenter.x;
						fs << "relaCenter_Y" << templ.relaCenter.y;
						fs << "initEdgeNum" << templ.InitEdgeNum;
						fs << "shape_scale" << templ.shape_info.scale;
						fs << "shape_angle" << templ.shape_info.angle;
						fs << "feature_size" << (int)templ.features.size();
						fs << "features"
							<< "[";
						{
							for (int k = 0; k < (int)templ.features.size(); k++) {

								auto feat = templ.features[k];
								fs << "[:" << feat.x << feat.y << feat.lbl << "]";
							}
						}
						fs << "]";
						fs << "}";
					}
				}
				fs << "]";
				fs << "}";
			}
		}
		fs << "]";
	}

	void ShapeMatching::LoadModel() {

		ClearModel();
		string model_name = model_root_ + class_name_ + SM_MODEL_SUFFUX;
		FileStorage fs(model_name, FileStorage::READ);
		assert(fs.isOpened() && "load model failed.");
		FileNode fn = fs.root();
		angle_range_.begin = fn["angle_range_bgin"];
		angle_range_.end = fn["angle_range_end"];
		angle_range_.step = fn["angle_range_step"];
		scale_range_.begin = fn["scale_range_bgin"];
		scale_range_.end = fn["scale_range_end"];
		scale_range_.step = fn["scale_range_step"];
		FileNode tps_fn = fn["templates"];
		FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
		for (; tps_it != tps_it_end; ++tps_it)
		{
			int template_id = (*tps_it)["template_id"];
			FileNode pyrds_fn = (*tps_it)["template_pyrds"];
			FileNodeIterator pyrd_it = pyrds_fn.begin(), pyrd_it_end = pyrds_fn.end();
			int pl = 0;
			for (; pyrd_it != pyrd_it_end; ++pyrd_it)
			{
				FileNode pyrd_fn = (*pyrd_it);
				Template templ;
				templ.id = pyrd_fn["id"];
				templ.pyramid_level = pyrd_fn["pyramid_level"];
				templ.is_valid = pyrd_fn["is_valid"];
				templ.x = pyrd_fn["x"];
				templ.y = pyrd_fn["y"];
				templ.w = pyrd_fn["w"];
				templ.h = pyrd_fn["h"];
				templ.InitEdgeNum = pyrd_fn["initEdgeNum"];
				templ.relaCenter.x = pyrd_fn["relaCenter_X"];
				templ.relaCenter.y = pyrd_fn["relaCenter_Y"];
				templ.shape_info.scale = pyrd_fn["shape_scale"];
				templ.shape_info.angle = pyrd_fn["shape_angle"];
				FileNode features_fn = pyrd_fn["features"];
				FileNodeIterator feature_it = features_fn.begin(), feature_it_end = features_fn.end();
				for (; feature_it != feature_it_end; ++feature_it)
				{
					FileNode feature_fn = (*feature_it);
					FileNodeIterator feature_info = feature_fn.begin();
					Feature feat;
					feature_info >> feat.x >> feat.y >> feat.lbl;
					templ.features.push_back(feat);
				}
				templ_all_[pl].push_back(templ);
				pl++;
			}
		}

		LoadRegion8Idxes();
	}

	void ShapeMatching::ClearModel() {

		for (int i = 0; i < PyramidLevel_TabooUse; i++) {

			templ_all_[i].clear();
		}
	}

	//初始化参数
	void ShapeMatching::InitMatchParameter(float score_thresh, float overlap, float mag_thresh, float greediness, int T, int top_k, MatchingStrategy strategy) {

		score_thresh_ = score_thresh;
		overlap_ = overlap;
		mag_thresh_ = mag_thresh;
		greediness_ = greediness;
		T_ = T;
		top_k_ = top_k;
		strategy_ = strategy;
	}

	//获取目标图像的全部金字塔图像
	void ShapeMatching::GetAllPyramidLevelValidSource(cv::Mat &source, PyramidLevel pyrd_level) {

		sources_.clear();
		for (int pl = 0; pl <= pyrd_level; pl++) {

			Mat source_pyrd;
			if (pl == 0) source_pyrd = source;
			else pyrDown(source, source_pyrd, Size(source.cols >> 1, source.rows >> 1));
			source = source_pyrd;
			sources_.push_back(source_pyrd);
		}
	}

	//获取前k个高分匹配
	vector<Match> ShapeMatching::GetTopKMatches(vector<Match> matches) {

		vector<Match> top_k_matches;
		top_k_matches.clear();
		if (top_k_ > 0 && (top_k_ < matches.size()) && (matches.size() > 0)) {

			int k = 0;
			top_k_matches.push_back(matches[0]);
			for (int m = 1; m < matches.size(); m++) {

				if (matches[m].similarity < matches[m - 1].similarity) {

					++k;
					if (k >= top_k_) break;
				}
				top_k_matches.push_back(matches[m]);
			}
		}
		else
		{
			top_k_matches = matches;
		}
		return top_k_matches;
	}

	//做非极大值抑制，进行重叠区域的筛选
	vector<Match> ShapeMatching::DoNmsMatches(vector<Match> matches, PyramidLevel pl, float overlap) {

		vector<Rect> boxes; boxes.clear();
		vector<float> scores; scores.clear();
		vector<int> indices; indices.clear();
		for (int m = 0; m < matches.size(); m++) {

			auto templ = templ_all_[pl][matches[m].template_id];
			Rect box = Rect(matches[m].x, matches[m].y, templ.w, templ.h);
			boxes.insert(boxes.end(), box);
			scores.insert(scores.end(), matches[m].similarity);
		}
		cv_dnn_nms::NMSBoxes(boxes, scores, overlap, overlap, indices);
		vector<Match> final_matches; final_matches.clear();
		for (auto index : indices) {

			final_matches.push_back(matches[index]);
		}
		return final_matches;
	}

	//量化180 的匹配
	vector<Match> ShapeMatching::MatchingPyrd180(Mat src, PyramidLevel pl, vector<int> region_idxes) {

		pl = PyramidLevel(pl + 8);
		vector<Match> matches; matches.clear();
		Mat angle, quantized_angle, mag;
		//量化目标图像
		QuantifyEdge(src, angle, quantized_angle, mag, mag_thresh_, true);
#pragma omp parallel 
		{
			int tlsz = region_idxes.empty() ? ((int)templ_all_[pl].size()) : ((int)region_idxes.size());
#pragma omp for nowait
			for (int t = 0; t < tlsz; t++) {

				Template templ = region_idxes.empty() ? (templ_all_[pl][t]) : (templ_all_[pl][region_idxes[t]]);
				for (int r = 0; r < quantized_angle.rows - templ.h; r++) {

					for (int c = 0; c < quantized_angle.cols - templ.w; c++) {

						int fsz = (int)templ.features.size();
						float partial_sum = 0.f;
						bool valid = true;
						for (int f = 0; f < fsz; f++) {

							Feature feat = templ.features[f];
							int sidx = quantized_angle.ptr<unsigned char>(r + feat.y)[c + feat.x];
							int tidx = feat.lbl;
							if (sidx != 255) {

								partial_sum += score_table_[sidx][tidx];
							}
							if (partial_sum + (fsz - f) * greediness_ < score_thresh_ * fsz) {

								valid = false;
								break;
							}
						}
						if (valid) {

							float score = partial_sum / fsz;
							if (score >= score_thresh_) {

								Match match;
								match.x = c + templ.w / 2;
								match.y = r + templ.h / 2;
								match.similarity = score;
								match.template_id = templ.id;
#pragma omp critical
								matches.insert(matches.end(), match);
							}
						}

					}
				}
			}
		}
		matches = DoNmsMatches(matches, pl, overlap_);
		return matches;
	}
	//		//量化180 的匹配
	//	vector<Match> ShapeMatching::MatchingPyrd180(Mat src, PyramidLevel pl, vector<int> region_idxes) {
	//
	//		pl = PyramidLevel(pl + 8);
	//		vector<Match> matches; matches.clear();
	//		Mat angle, quantized_angle, mag;
	//		//量化目标图像
	//		QuantifyEdge(src, angle, quantized_angle, mag, mag_thresh_, true);
	//#pragma omp parallel 
	//		{
	//			int tlsz = region_idxes.empty() ? ((int)templ_all_[pl].size()) : ((int)region_idxes.size());
	//#pragma omp for nowait
	//			for (int t = 0; t < tlsz; t++) {
	//
	//				Template templ = region_idxes.empty() ? (templ_all_[pl][t]) : (templ_all_[pl][region_idxes[t]]);
	//				for (int r = 0; r < quantized_angle.rows - templ.h; r++) {
	//
	//					for (int c = 0; c < quantized_angle.cols - templ.w; c++) {
	//
	//						int fsz = (int)templ.features.size();
	//						float partial_sum = 0.f;
	//						double RightNum = 0.0;
	//						double WrongNum = 0.0;
	//						bool valid = true;
	//						for (int f = 0; f < fsz; f++) {
	//
	//							Feature feat = templ.features[f];
	//							int sidx = quantized_angle.ptr<unsigned char>(r + feat.y)[c + feat.x];
	//							int tidx = feat.lbl;
	//							if (sidx != 255) {
	//
	//								double score= score_table_[sidx][tidx];
	//								if ((score > 0.8)&&(score<0.9))RightNum++;
	//								else if (score > 0.9)RightNum += 2;
	//								else {
	//									WrongNum++;
	//								}
	//							}
	//							if (WrongNum>(0.5 * (fsz))) {
	//
	//								valid = false;
	//								break;
	//							}
	//						}
	//						if (valid) {
	//
	//							float score = float(RightNum / (fsz));
	//							if (score >= score_thresh_) {
	//
	//								Match match;
	//								match.x = c;
	//								match.y = r;
	//								match.similarity = score;
	//								match.template_id = templ.id;
	//								match.feature_size = fsz;
	//								match.RightNum = RightNum;
	//#pragma omp critical
	//								matches.insert(matches.end(), match);
	//							}
	//						}
	//
	//					}
	//				}
	//			}
	//		}
	//		matches = DoNmsMatches(matches, pl, overlap_);
	//		return matches;
	//	}

		//量化8方向匹配
	vector<Match> ShapeMatching::MatchingPyrd8(Mat src, PyramidLevel pl, bool isTopLevel, vector<int> region_idxes) {

		//筛选目标图像的搜索区域
		cv::Mat flagImage = filterSource(src);

		vector<Match> matches; matches.clear();
		Mat angle, quantized_angle, mag;
		QuantifyEdge(src, angle, quantized_angle, mag, mag_thresh_, false);

		Mat spread_angle;
		//梯度扩散
		Spread(quantized_angle, spread_angle, T_);
		vector<Mat> response_maps;
		//计算响应图
		ComputeResponseMaps(spread_angle, response_maps);
#pragma omp parallel 
		{
			//此层金字塔的所有图像数量
			int tlsz = region_idxes.empty() ? ((int)templ_all_[pl].size()) : ((int)region_idxes.size());
#pragma omp for nowait
			for (int t = 0; t < tlsz; t++) {

				Template templ = region_idxes.empty() ? (templ_all_[pl][t]) : (templ_all_[pl][region_idxes[t]]);
				for (int r = 0; r < quantized_angle.rows - templ.h; r += T_) {
					unsigned char*flagImg_ptr = flagImage.ptr<uchar>(r);
					for (int c = 0; c < quantized_angle.cols - templ.w; c += T_) {
						if ((flagImg_ptr + templ.h / 2 * flagImage.step)[c + templ.w / 2] == 0)continue;
						//if (flagImage.at<uchar>(r+templ.h/2,c+templ.w/2) == 0)continue;
						else {
							int fsz = (int)templ.features.size();
							int partial_sum = 0;
							bool valid = true;
							for (int f = 0; f < fsz; f++) {

								Feature feat = templ.features[f];
								int label = feat.lbl;
								//模板图像的点在原图像上点的位置
								partial_sum +=
									response_maps[label].ptr<unsigned char>(r + feat.y)[c + feat.x];//利用幅角内积进行求相似度
								if (partial_sum + (fsz - f) * greediness_ * 100 < score_thresh_ * fsz * 100) {

									valid = false;
									break;
								}
							}
							if (valid) {

								float score = partial_sum / (100.f * fsz);
								if (score >= score_thresh_) {

									Match match;
									match.x = c;
									match.y = r;
									match.similarity = score;
									match.template_id = templ.id;
#pragma omp critical
									matches.insert(matches.end(), match);
								}
							}
						}

					}
				}
			}
		}
		matches = DoNmsMatches(matches, pl, overlap_);
		return matches;

	}

	//		//量化8方向匹配
	//	vector<Match> ShapeMatching::MatchingPyrd8(Mat src, PyramidLevel pl, bool isTopLevel,vector<int> region_idxes) {
	//
	//		vector<Match> matches; matches.clear();
	//		Mat angle, quantized_angle, mag;
	//		QuantifyEdge(src, angle, quantized_angle, mag, mag_thresh_, false);
	//
	//		//判断是否是顶层
	//		if (isTopLevel)
	//		{
	//			cv::Mat edgeImg = SourceExtractTemplate(angle, quantized_angle, mag, weak_thresh_, strong_thresh_);
	//
	//			//计算边缘图像的积分图
	//			cv::Mat Integral = Integral_row(edgeImg);
	//
	//			Template templ = region_idxes.empty() ? (templ_all_[pl][0]) : (templ_all_[pl][region_idxes[0]]);
	//			cv::Mat isMatching = cv::Mat::zeros(src.size(), CV_8UC1);
	//#pragma omp parallel for
	//			for (int r = 0; r < edgeImg.rows - templ.h; r++)
	//			{
	//				float* integral_ptr = Integral.ptr<float>(r);
	//				unsigned char *match_ptr = isMatching.ptr<uchar>(r);
	//				float *matchEnd_ptr = Integral.ptr<float>(r + templ.h);
	//				for (int c = 0; c < edgeImg.cols - templ.w; c++)
	//				{
	//					int num = 0;
	//					//num = Integral.at<float>(r, c) + Integral.at<float>(r + templ.h, c + templ.w)
	//					//	- Integral.at<float>(r + templ.h, c) - Integral.at<float>(r, c + templ.w);
	//					num = matchEnd_ptr[c + templ.w] + integral_ptr[c] - integral_ptr[c + templ.w]
	//						- matchEnd_ptr[c];
	//
	//					if (num > double(templ.InitEdgeNum)*0.7)
	//					{
	//						match_ptr[c] = 255;
	//					}
	//				}
	//			}
	//
	//			Mat spread_angle;
	//			//梯度扩散
	//			Spread(quantized_angle, spread_angle, T_);
	//			vector<Mat> response_maps;
	//			//计算响应图
	//			ComputeResponseMaps(spread_angle, response_maps);
	//#pragma omp parallel 
	//			{
	//				//此层金字塔的所有图像数量
	//				int tlsz = region_idxes.empty() ? ((int)templ_all_[pl].size()) : ((int)region_idxes.size());
	//#pragma omp for nowait
	//				for (int t = 0; t < tlsz; t++) {
	//
	//					Template templ = region_idxes.empty() ? (templ_all_[pl][t]) : (templ_all_[pl][region_idxes[t]]);
	//					for (int r = 0; r < quantized_angle.rows - templ.h; r += T_) {
	//
	//						for (int c = 0; c < quantized_angle.cols - templ.w; c += T_) {
	//
	//							int fsz = (int)templ.features.size();
	//							int partial_sum = 0;
	//							bool valid = true;
	//							double RightNum = 0;		//用来存储正确的点数量
	//							double WrongNum = 0;		//用来存储错误的点数量
	//							for (int f = 0; f < fsz; f++) {
	//
	//								Feature feat = templ.features[f];
	//								int label = feat.lbl;
	//								//模板图像的点在原图像上点的位置
	//								double score = response_maps[label].ptr<unsigned char>(r + feat.y)[c + feat.x];//利用幅角内积进行求相似度
	//								//partial_sum +=
	//								//	response_maps[label].ptr<unsigned char>(r + feat.y)[c + feat.x];//利用幅角内积进行求相似度
	//								//if (partial_sum + (fsz - f) * greediness_ * 100 < score_thresh_ * fsz * 100) {
	//
	//								//	valid = false;
	//								//	break;
	//								//}
	//								if ((score > 0.8)&&(score<0.9))
	//								{
	//									RightNum++;
	//								}
	//								else if (score > 0.9)RightNum+=2;
	//								else {
	//									WrongNum++;
	//								}
	//								if (WrongNum > (0.5*(fsz)))
	//								{
	//									valid = false;
	//									break;
	//								}
	//							}
	//							if (valid) {
	//
	//								float score = RightNum / double(fsz);
	//								if (score >= score_thresh_) {
	//
	//									Match match;
	//									match.x = c;
	//									match.y = r;
	//									match.similarity = score;
	//									match.template_id = templ.id;
	//									match.RightNum = RightNum;
	//									match.feature_size = fsz;
	//#pragma omp critical
	//									matches.insert(matches.end(), match);
	//								}
	//							}
	//						}
	//					}
	//				}
	//			}
	//			matches = DoNmsMatches(matches, pl, overlap_);
	//			return matches;
	//		}
	//
	//		else {
	//		Mat spread_angle;
	//		//梯度扩散
	//		Spread(quantized_angle, spread_angle, T_);
	//		vector<Mat> response_maps;
	//		//计算响应图
	//		ComputeResponseMaps(spread_angle, response_maps);
	//#pragma omp parallel 
	//		{
	//			//此层金字塔的所有图像数量
	//			int tlsz = region_idxes.empty() ? ((int)templ_all_[pl].size()) : ((int)region_idxes.size());
	//#pragma omp for nowait
	//			for (int t = 0; t < tlsz; t++) {
	//
	//				Template templ = region_idxes.empty() ? (templ_all_[pl][t]) : (templ_all_[pl][region_idxes[t]]);
	//				for (int r = 0; r < quantized_angle.rows - templ.h; r += T_) {
	//
	//					for (int c = 0; c < quantized_angle.cols - templ.w; c += T_) {
	//
	//						int fsz = (int)templ.features.size();
	//						int partial_sum = 0;
	//						bool valid = true;
	//						double RightNum = 0;		//用来存储正确的点数量
	//						double WrongNum = 0;		//用来存储错误的点数量
	//						for (int f = 0; f < fsz; f++) {
	//
	//							Feature feat = templ.features[f];
	//							int label = feat.lbl;
	//							//模板图像的点在原图像上点的位置
	//							double score = response_maps[label].ptr<unsigned char>(r + feat.y)[c + feat.x];//利用幅角内积进行求相似度
	//							//partial_sum +=
	//							//	response_maps[label].ptr<unsigned char>(r + feat.y)[c + feat.x];//利用幅角内积进行求相似度
	//							//if (partial_sum + (fsz - f) * greediness_ * 100 < score_thresh_ * fsz * 100) {
	//
	//							//	valid = false;
	//							//	break;
	//							//}
	//							if ((score > 0.8) && (score < 0.9))
	//							{
	//								RightNum++;
	//							}
	//							else if (score > 0.9)RightNum += 2;
	//							else {
	//								WrongNum++;
	//							}
	//							if (WrongNum > (0.5*(fsz)))
	//							{
	//								valid = false;
	//								break;
	//							}
	//						}
	//						if (valid) {
	//
	//							float score = RightNum / double(fsz);
	//							if (score >= score_thresh_) {
	//
	//								Match match;
	//								match.x = c;
	//								match.y = r;
	//								match.similarity = score;
	//								match.template_id = templ.id;
	//								match.RightNum = RightNum;
	//								match.feature_size = fsz;
	//#pragma omp critical
	//								matches.insert(matches.end(), match);
	//							}
	//						}
	//					}
	//				}
	//			}
	//		}
	//		matches = DoNmsMatches(matches, pl, overlap_);
	//		return matches;
	//	}
	//}

		//梯度扩散
	void ShapeMatching::Spread(const Mat quantized_angle, Mat &spread_angle, int T) {

		spread_angle = Mat::zeros(quantized_angle.size(), CV_8U);
		int cols = quantized_angle.cols;
		int rows = quantized_angle.rows;
		int half_T = 0;
		if (T != 1) half_T = T / 2;
#pragma omp parallel for
		for (int r = half_T; r < rows - half_T; r++) {

			for (int c = half_T; c < cols - half_T; c++) {

				for (int i = -half_T; i <= half_T; i++) {

					for (int j = -half_T; j <= half_T; j++) {

						unsigned char shift_bits =
							quantized_angle.ptr<unsigned char>(r + i)[c + j];
						if (shift_bits < 8) {

							spread_angle.ptr<unsigned char>(r)[c] |=
								(unsigned char)(1 << shift_bits);
						}
					}
				}
			}
		}
	}

	void ShapeMatching::ComputeResponseMaps(const Mat spread_angle, vector<Mat> &response_maps) {

		response_maps.clear();
		for (int i = 0; i < 8; i++) {

			Mat rm;
			rm.create(spread_angle.size(), CV_8U);
			response_maps.push_back(rm);
		}
		int cols = spread_angle.cols;
		int rows = spread_angle.rows;
#pragma omp parallel for
		for (int i = 0; i < 8; i++) {

			for (int r = 0; r < rows; r++) {

				for (int c = 0; c < cols; c++) {

					response_maps[i].ptr<unsigned char>(r)[c] =
						score_table_8map_[i][spread_angle.ptr<unsigned char>(r)[c]];
				}
			}
		}
	}

	//获取在下层金字塔的起始位置
	bool ShapeMatching::CalcPyUpRoiAndStartPoint(PyramidLevel cur_pl, PyramidLevel obj_pl, Match match,
		Mat &r, Point &p, bool is_padding) {

		auto templ = templ_all_[cur_pl][match.template_id];
		int padding = 0;

		//判断是否需要进行填充
		if (is_padding) {

			int min_side = std::min(templ.w, templ.h);
			//ceil() 函数返回大于或等于给定参数的最小可能整数值
			int diagonal_line_length = (int)ceil(sqrt(templ.w*templ.w + templ.h*templ.h));//对角线长度
			padding = diagonal_line_length - min_side;
		}
		int err_pl = cur_pl - obj_pl;
		int T = 2 * T_;
		int extend_pixel = 1;
		cv::Point bp, ep;
		int multiple = (1 << err_pl);		//1向左移动0位，结果还是1
		match.x -= (T + padding) / 2;
		match.y -= (T + padding) / 2;
		templ.w += (T + padding);
		templ.h += (T + padding);

		//bp:左上角点   ep:右下角点
		bp.x = (match.x - extend_pixel) * multiple;
		bp.y = (match.y - extend_pixel) * multiple;
		ep.x = (match.x + templ.w + extend_pixel) * multiple;
		ep.y = (match.y + templ.h + extend_pixel) * multiple;

		//越界处理
		if (bp.x < 0)
		{
			ep.x = ep.x + (0 - bp.x);
			bp.x = 0;
		}
		if (bp.y < 0)
		{
			ep.y = ep.y + (0 - bp.y);
			bp.y = 0;
		}
		if (ep.x < 0) ep.x = 0;
		if (ep.y < 0) ep.y = 0;
		if (bp.x >= sources_[obj_pl].cols) bp.x = sources_[obj_pl].cols - 1;
		if (bp.y >= sources_[obj_pl].rows) bp.y = sources_[obj_pl].rows - 1;
		if (ep.x >= sources_[obj_pl].cols)
		{
			bp.x = bp.x - (ep.x - sources_[obj_pl].cols + 1);
			ep.x = sources_[obj_pl].cols - 1;
		}
		if (ep.y >= sources_[obj_pl].rows)
		{
			bp.y = bp.y - (ep.y - sources_[obj_pl].rows + 1);
			ep.y = sources_[obj_pl].rows - 1;
		}


		//如果两点不重合
		if (bp.x != ep.x || bp.y != ep.y) {

			Rect rect = Rect(bp, ep);
			Mat roi(sources_[obj_pl], rect);
			r = roi;
			p = bp;
			return true;
		}
		else
		{
			return false;
		}
	}

	//计算区域的索引
	void ShapeMatching::CalcRegionIndexes(vector<int> &region_idxes, Match match, MatchingStrategy strategy) {

		region_idxes.clear();
		Template templ = templ_all_[PyramidLevel_0][match.template_id];
		float match_agl = templ.shape_info.angle;//最佳匹配的角度
		float match_sal = templ.shape_info.scale;//最佳匹配的尺度
		int angle_region = (int)((angle_range_.end - angle_range_.begin) / angle_range_.step) + 1;
		int scale_region = (int)((scale_range_.end - scale_range_.begin) / scale_range_.step) + 1;
		if (strategy <= Strategy_Middling) {

			if (match_agl < 0.f) match_agl += 360.f;
			int key = (int)floor(match_agl / 22.5f);		//对浮点数进行向下取整操作
			float left_agl = match_agl - key * 22.5f;
			for (int ar = 0; ar < angle_region; ar++) {

				float cur_agl = templ_all_[PyramidLevel_0][ar].shape_info.angle;
				if (cur_agl < 0.f) cur_agl += 360.f;
				int k = key;
				if (cur_agl >= AngleRegionTable[k][0] && cur_agl < AngleRegionTable[k][1]) {

					for (int sr = 0; sr < scale_region; sr++) {

						region_idxes.push_back(ar + sr * angle_region);
					}
				}
				if (strategy == Strategy_Accurate) {

					if (left_agl < 11.25f) {

						k = key - 1;
						if (k < 0) k = 15;
						if (cur_agl >= AngleRegionTable[k][0] && cur_agl < AngleRegionTable[k][1]) {

							for (int sr = 0; sr < scale_region; sr++) {

								region_idxes.push_back(ar + sr * angle_region);
							}
						}
					}
					else
					{
						k = key + 1;
						if (k > 15) k = 0;
						if (cur_agl >= AngleRegionTable[k][0] && cur_agl < AngleRegionTable[k][1]) {

							for (int sr = 0; sr < scale_region; sr++) {

								region_idxes.push_back(ar + sr * angle_region);
							}
						}
					}
				}
			}
		}
		else if (strategy == Strategy_Rough) {

			float err_range = 3.f;
			for (int ar = 0; ar < angle_region; ar++) {
				//角度在最佳匹配角度左右三个步长范围内
				float cur_agl = templ_all_[PyramidLevel_0][ar].shape_info.angle;
				if (cur_agl >= (match_agl - angle_range_.step * err_range) &&
					cur_agl <= (match_agl + angle_range_.step * err_range)) {

					for (int sr = 0; sr < scale_region; sr++) {
						//尺度在最佳匹配尺度上下三个步长范围内
						float cur_sal = templ_all_[PyramidLevel_0][ar + sr * angle_region].shape_info.scale;
						if (cur_sal >= (match_sal - scale_range_.step * err_range) &&
							cur_sal <= (match_sal + scale_range_.step * err_range)) {

							region_idxes.push_back(ar + sr * angle_region);
						}
					}
				}
			}
		}
	}

	vector<Match> ShapeMatching::ReconfirmMatches(vector<Match> matches, PyramidLevel pl) {

		vector<Match> rf_matches;
		rf_matches.clear();
		for (int i = 0; i < matches.size(); i++) {

			Mat roi;
			Point sp;
			CalcPyUpRoiAndStartPoint(pl, pl, matches[i], roi, sp, true);
			vector<int> region_idxes;
			CalcRegionIndexes(region_idxes, matches[i], Strategy_Accurate);
			auto tmp_matches = MatchingPyrd8(roi, pl, false, region_idxes);
			if (tmp_matches.size() > 0) {

				tmp_matches[0].x += sp.x;
				tmp_matches[0].y += sp.y;
				rf_matches.push_back(tmp_matches[0]);
			}
		}
		rf_matches = DoNmsMatches(rf_matches, pl, overlap_);
		return rf_matches;
	}

	//最后匹配，量化的180
	vector<Match> ShapeMatching::MatchingFinal(vector<Match> matches, PyramidLevel pl) {

		vector<Match> final_matches;
		final_matches.clear();
		for (int i = 0; i < matches.size(); i++) {

			Mat roi;
			Point sp;
			CalcPyUpRoiAndStartPoint(pl, PyramidLevel_0, matches[i], roi, sp, false);
			vector<int> region_idxes;
			CalcRegionIndexes(region_idxes, matches[i], strategy_);
			auto tmp_matches = MatchingPyrd180(roi, PyramidLevel_0, region_idxes);
			if (tmp_matches.size() > 0) {

				tmp_matches[0].x += sp.x;
				tmp_matches[0].y += sp.y;
				final_matches.push_back(tmp_matches[0]);
			}
		}
		final_matches = DoNmsMatches(final_matches, pl, overlap_);
		return final_matches;
	}


	float KCluster::getDistance(PT point, PT center)
	{
		return sqrt(pow(point.x - center.x, 2) + pow(point.y - center.y, 2)
			+ pow(point.mag - center.mag, 2) + pow(point.ang - center.ang, 2));
	}

	//做一个聚类，将模板图像的边缘点进行分块
	void KCluster::doCluster(cv::Mat &model, cv::Mat mag, cv::Mat angle, std::vector<cv::Point2f> points, int k)
	{
		Mat img = Mat::zeros(model.rows, model.cols, CV_8UC3);

		cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
		cv::normalize(angle, angle, 0, 1, cv::NORM_MINMAX);
		std::vector<PT>vPT;
		for (std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
		{
			PT pt;
			pt.x = (*it).x;
			pt.y = (*it).y;
			pt.mag = mag.at<float>((*it).y, (*it).x);
			pt.ang = angle.at<float>((*it).y, (*it).x);

			vPT.push_back(pt);
		}

		//kmeans

		//使用前几个坐标初始化中心点
		std::vector<PT> center;
		for (int i = 0; i < k; ++i)
		{
			center.push_back(vPT[i]);
		}

		//初始化label
		vector<int> label(points.size(), -1);

		//迭代
		for (int iter = 0; iter < ITER; ++iter)
		{
			//根据与中心点的距离，对每一个样本点进行聚类
			for (int pointNum = 0; pointNum < points.size(); ++pointNum)
			{
				float distance = FLT_MAX;

				for (int cluster = 0; cluster < k; ++cluster)
				{
					float temp_distance = getDistance(vPT[pointNum], center[cluster]);
					if (temp_distance < distance)
					{
						distance = temp_distance;

						label[pointNum] = cluster;
					}
				}
			}

			//根据聚类结果，计算坐标均值，更新中心点坐标
			for (int cluster = 0; cluster < k; cluster++)
			{
				int count = 0;
				int sum_x = 0;
				int sum_y = 0;
				for (int pointNum = 0; pointNum < points.size(); pointNum++)
				{
					if (label[pointNum] == cluster)
					{
						count++;
						sum_x += points[pointNum].x;
						sum_y += points[pointNum].y;
					}
				}
				center[cluster].x = sum_x / count;
				center[cluster].y = sum_y / count;
			}
		}


		for (int i = 0; i < points.size(); i++)
		{
			if (label[i] == 0)
			{
				circle(img, points[i], 1, Scalar(255, 0, 0), FILLED, LINE_AA);
			}
			if (label[i] == 1)
			{
				circle(img, points[i], 1, Scalar(0, 255, 0), FILLED, LINE_AA);
			}
			if (label[i] == 2)
			{
				circle(img, points[i], 1, Scalar(0, 0, 255), FILLED, LINE_AA);
			}
			if (label[i] == 3)
			{
				circle(img, points[i], 1, Scalar(255, 0, 255), FILLED, LINE_AA);
			}
		}

		namedWindow("img", WINDOW_AUTOSIZE);
		imshow("img", img);
		waitKey(0);
	}

	//双阈值设置提取边缘，canny算子
	cv::Mat ShapeMatching::SourceExtractTemplate(Mat angle, Mat quantized_angle, Mat mag,
		float weak_thresh, float strong_thresh, Mat mask) {

		cv::Mat edgeImg = cv::Mat::zeros(angle.size(), CV_8UC1);

		Mat local_angle = Mat(angle.size(), angle.type());
		for (int r = 0; r < angle.rows; ++r) {


			float *angle_ptr = angle.ptr<float>(r);
			float *local_angle_ptr = local_angle.ptr<float>(r);
			for (int c = 0; c < angle.cols; ++c) {

				float dir = angle_ptr[c];
				if ((dir > 0. && dir < 22.5) || (dir > 157.5 && dir < 202.5) || (dir > 337.5 && dir < 360.))
					local_angle_ptr[c] = 0.f;
				else if ((dir > 22.5 && dir < 67.5) || (dir > 202.5 && dir < 247.5))
					local_angle_ptr[c] = 45.f;
				else if ((dir > 67.5 && dir < 112.5) || (dir > 247.5 && dir < 292.5))
					local_angle_ptr[c] = 90.f;
				else if ((dir > 112.5 && dir < 157.5) || (dir > 292.5 && dir < 337.5))
					local_angle_ptr[c] = 135.f;
				else
					local_angle_ptr[c] = 0.f;
			}
		}

		bool no_mask = mask.empty();
		float weak_sq = weak_thresh * weak_thresh;
		float strong_sq = strong_thresh * strong_thresh;
		float pre_grad, lst_grad;
		for (int r = 1; r < mag.rows - 1; ++r)
		{
			unsigned char *edge_ptr = edgeImg.ptr<unsigned char>(r);

			const unsigned char *mask_ptr = no_mask ? NULL : mask.ptr<unsigned char>(r);
			const float* pre_ptr = mag.ptr<float>(r - 1);
			const float* cur_ptr = mag.ptr<float>(r);
			const float* lst_ptr = mag.ptr<float>(r + 1);
			float *local_angle_ptr = local_angle.ptr<float>(r);

			for (int c = 1; c < mag.cols - 1; ++c)
			{
				if (no_mask || mask_ptr[c])
				{
					switch ((int)local_angle_ptr[c]) {

					case 0:
						pre_grad = cur_ptr[c - 1];
						lst_grad = cur_ptr[c + 1];
						break;
					case 45:
						pre_grad = pre_ptr[c + 1];
						lst_grad = lst_ptr[c - 1];
						break;
					case 90:
						pre_grad = pre_ptr[c];
						lst_grad = lst_ptr[c];
						break;
					case 135:
						pre_grad = pre_ptr[c - 1];
						lst_grad = lst_ptr[c + 1];
						break;
					}
					if ((cur_ptr[c] > pre_grad) && (cur_ptr[c] > lst_grad)) {

						float score = cur_ptr[c];
						bool validity = false;
						if (score >= weak_sq) {

							if (score >= strong_sq) {

								validity = true;
							}
							else {

								if (((pre_ptr[c - 1]) >= strong_sq) ||
									((pre_ptr[c]) >= strong_sq) ||
									((pre_ptr[c + 1]) >= strong_sq) ||
									((cur_ptr[c - 1]) >= strong_sq) ||
									((cur_ptr[c + 1]) >= strong_sq) ||
									((lst_ptr[c - 1]) >= strong_sq) ||
									((lst_ptr[c]) >= strong_sq) ||
									((lst_ptr[c + 1]) >= strong_sq))
								{
									validity = true;
								}
							}
						}
						if (validity == true &&
							quantized_angle.at<unsigned char>(r, c) != 255) {


							edge_ptr[c] = 1;
							//Candidate cd;
							//cd.score = score;
							//cd.feature.x = c;
							//cd.feature.y = r;
							//cd.feature.lbl = quantized_angle.at<unsigned char>(r, c);
							//candidates.push_back(cd);
						}
					}

				}
			}
		}
		return edgeImg;
	}

	//快速积分图的计算
	cv::Mat ShapeMatching::Integral_row(cv::Mat src)
	{
		cv::Mat integal_out;
		src.convertTo(integal_out, CV_32F);
		float *p = integal_out.ptr<float>(0);
		for (int i = 1; i < src.cols; i++)   //计算第一行像素点的积分值
		{
			p[i] += p[i - 1];
		}
		float *p1;
		//#pragma omp parallel for
		for (int i = 1; i < src.rows; i++)   //从第二行开始，第i行
		{
			float sum = 0.0;   //累加和变量
			p = integal_out.ptr<float>(i);
			p1 = integal_out.ptr<float>(i - 1);
			for (int j = 0; j < src.cols; j++)   //第j列
			{
				sum += p[j];      //累加当前像素值
				p[j] = p1[j] + sum;    //计算当前像素点的积分值
			}
		}
		return integal_out;
	}

	//获取最适合匹配的通道
	cv::Mat ShapeMatching::rightChannel(cv::Mat &image)
	{
		cv::Mat resultChannel;

		vector<Mat> channels;
		split(image, channels);

		Mat blueChannel = channels[0];
		Mat greenChannel = channels[1];
		Mat redChannel = channels[2];

		// 计算积分图
		Mat blueIntegralImage, greenIntegralImage, redIntegralImage;
		integral(blueChannel, blueIntegralImage, CV_64F);
		integral(greenChannel, greenIntegralImage, CV_64F);
		integral(redChannel, redIntegralImage, CV_64F);

		//计算积分图的最后一个元素
		double bluePixels = blueIntegralImage.at<double>(blueIntegralImage.rows - 1, blueIntegralImage.cols - 1);
		double greenPixels = greenIntegralImage.at<double>(greenIntegralImage.rows - 1, greenIntegralImage.cols - 1);
		double redPixels = redIntegralImage.at<double>(redIntegralImage.rows - 1, redIntegralImage.cols - 1);

		if (bluePixels > greenPixels && bluePixels > redPixels)
		{
			resultChannel = blueChannel;
		}
		else if (greenPixels > bluePixels && greenPixels > redPixels)
		{
			resultChannel = greenChannel;
		}
		else
		{
			resultChannel = redChannel;
		}
		return resultChannel;
	}

	//获取合适的HSV通道
	HSV_I ShapeMatching::getHSVinfo(cv::Mat &imageRGB, int channelIndex, bool flag)
	{


		HSV_I hsvImage;
		// 将图像转换为 HSV 颜色空间
		Mat hsv_image;
		cvtColor(imageRGB, hsv_image, COLOR_BGR2HSV);

		cv::Rect rect = cv::Rect(10, 10, imageRGB.cols - 20, imageRGB.rows - 20);
		cv::Mat cut_hsv_image = hsv_image(rect);

		// 分离出三个通道
		//Mat h_channel, s_channel, v_channel;
		std::vector<Mat> channels;
		split(cut_hsv_image, channels);

		if (flag == 1)
		{
			hsvImage.channel = channelIndex;
			hsvImage.ChannelImage = channels[channelIndex];
			return hsvImage;
		}
		else {
			double H_pixel = sumPixel(channels[0]);
			double S_pixel = sumPixel(channels[1]);
			double V_pixel = sumPixel(channels[2]);

			//判断总和大小
			int max_id = (H_pixel > S_pixel) ? ((H_pixel > V_pixel) ? 0 : 2) : ((S_pixel > V_pixel) ? 1 : 2);

			std::vector<Mat> channels_01;
			split(hsv_image, channels_01);

			hsvImage.ChannelImage = channels_01[max_id];
			hsvImage.channel = max_id;


			return hsvImage;
		}


	}

	//计算图像像素值和的函数
	double ShapeMatching::sumPixel(cv::Mat &img)
	{
		double sum = 0;
		for (int r = 0; r < img.rows; r++)
		{
			uchar* img_ptr = img.ptr<uchar>(r);
			for (int c = 0; c < img.cols; c++)
			{
				sum += img_ptr[c];
			}
		}
		return sum;
	}

	//连通域分析消除孔洞
	void connected_domain(cv::Mat &img)
	{
		// Perform connected components analysis
		cv::Mat labels;
		int num_objects = cv::connectedComponents(img, labels);


	}

	//计算模板图像的mask
	cv::Mat ShapeMatching::makeMask(cv::Mat &templ)
	{
		cv::Mat mask = cv::Mat::zeros(templ.size(), templ.type());
		mask.setTo(255);

		cv::Mat templBin;
		cv::threshold(templ, templBin, 0, 255, THRESH_BINARY | THRESH_OTSU);


		//计算出全部轮廓图像，并计算出轮廓的最大外界矩形
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(templBin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		// 计算最大外接矩形
		int maxContour = 0;
		int maxContourIndex = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			if (contours[i].size() > maxContour)
			{
				maxContour = contours[i].size();
				maxContourIndex = i;
			}
		}

		cv::Rect maxRect = cv::boundingRect(contours[maxContourIndex]);

		cv::Mat mask01 = mask(maxRect);
		mask01.setTo(0);
		//腐蚀操作
		// 将图像取反
		cv::bitwise_not(mask, mask);
		cv::Mat m1;
		m1 = cv::getStructuringElement(0, cv::Size(12, 12));
		cv::Mat erodeM;
		cv::erode(mask, erodeM, m1, cv::Point(-1, -1), 5);
		cv::bitwise_not(erodeM, erodeM);

		return erodeM;

	}

	//获取资源图像的HSV通道图像
	cv::Mat ShapeMatching::sourceHSVChannel(cv::Mat &source, int index)
	{
		cv::Mat hsvImage;
		cv::cvtColor(source, hsvImage, COLOR_BGR2HSV);
		std::vector<cv::Mat>channels;
		split(hsvImage, channels);

		return channels[index];
	}

	//筛选资源图像的区域，加快匹配速度
	cv::Mat ShapeMatching::filterSource(cv::Mat &source)
	{
		cv::Mat sourceBin;
		cv::threshold(source, sourceBin, 0, 255, THRESH_BINARY | THRESH_OTSU);

		//闭合操作，将图像内部进行填充
		Mat element;
		element = getStructuringElement(MORPH_RECT, Size(15, 15));
		Mat closeImage;
		morphologyEx(sourceBin, closeImage, MORPH_CLOSE, element, cv::Point(-1, -1), 2);

		//对筛选出的前景进行腐蚀，减少搜索区域
		cv::Mat m1;
		m1 = cv::getStructuringElement(0, cv::Size(10, 10));
		cv::Mat resultImage;
		cv::erode(closeImage, resultImage, m1, cv::Point(-1, -1), 5);
		//return closeImage;
		return resultImage;
	}

} // end namespace kcg_matching

void  getFilePaths(std::vector<cv::String>& filepaths, cv::String filePath)  //对sort排序
{
	filepaths.clear();
	cout << "Read files from: " << filePath << endl;
	vector<cv::String> fn;
	cv::glob(filePath, fn, false);

	if (fn.size() == 0)
	{
		cout << "file " << filePath << " not  exits" << endl;

	}
	//prepare pair for sort 
	vector<pair<int, string>> v1;
	pair<int, string> p1;
	vector<cv::String >::iterator it_;
	for (it_ = fn.begin(); it_ != fn.end(); ++it_)
	{
		//1.获取不带路径的文件名,1.txt
		string::size_type iPos = (*it_).find_last_of('\\') + 1;
		string filename = (*it_).substr(iPos, (*it_).length() - iPos);
		//2.获取不带后缀的文件名,1
		string name = filename.substr(0, filename.rfind("."));
		//3.构建键和值的pair
		try {
			//防止文件夹中出现非整数的文件名导致的错误
			p1 = make_pair(stoi(name), (*it_).c_str());

		}
		catch (exception e)
		{
			cout << "Crushed -> " << e.what() << endl;
			//continue; 直接continue一样 
			it_ = fn.erase(it_);
			it_--; //erase函数的返回的是指向被删除元素的下一个元素的迭代器，所以执行erase（）后要把迭代器减1，指向前面一个
		}
		v1.emplace_back(p1);
	}
	//cout << "v1.sie(): " << v1.size()<<endl;
	sort(v1.begin(), v1.end(), [](auto a, auto b) {return a.first < b.first; });
	vector<pair<int, string> >::iterator it;
	for (it = v1.begin(); it != v1.end(); ++it)
	{
		//cout << it->first << endl;
		//cout << it->second << endl;

		filepaths.emplace_back(it->second);
	}
}

//获取最适合匹配的通道
int rightChannelNum(cv::Mat &image)
{
	int ChannelNum;

	vector<Mat> channels;
	split(image, channels);

	Mat blueChannel = channels[0];
	Mat greenChannel = channels[1];
	Mat redChannel = channels[2];

	// 计算积分图
	Mat blueIntegralImage, greenIntegralImage, redIntegralImage;
	integral(blueChannel, blueIntegralImage, CV_64F);
	integral(greenChannel, greenIntegralImage, CV_64F);
	integral(redChannel, redIntegralImage, CV_64F);

	//计算积分图的最后一个元素
	double bluePixels = blueIntegralImage.at<double>(blueIntegralImage.rows - 1, blueIntegralImage.cols - 1);
	double greenPixels = greenIntegralImage.at<double>(greenIntegralImage.rows - 1, greenIntegralImage.cols - 1);
	double redPixels = redIntegralImage.at<double>(redIntegralImage.rows - 1, redIntegralImage.cols - 1);

	if (bluePixels > greenPixels && bluePixels > redPixels)
	{
		ChannelNum = 0;
	}
	else if (greenPixels > bluePixels && greenPixels > redPixels)
	{
		ChannelNum = 1;
	}
	else
	{
		ChannelNum = 2;
	}
	return ChannelNum;
}

//获取通道图像
cv::Mat sourceChannel(cv::Mat &source, int channel)
{
	std::vector<cv::Mat> channels;
	split(source, channels);

	return channels[channel];
}

//获取资源图像的HSV通道图像
cv::Mat sourceHSVChannel(cv::Mat &source, int index)
{
	cv::Mat hsvImage;
	cv::cvtColor(source, hsvImage, COLOR_BGR2HSV);
	std::vector<cv::Mat>channels;
	split(hsvImage, channels);

	return channels[index];
}


