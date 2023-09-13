#pragma once
#pragma once

#ifndef _SM_MATCH_H_
#define _SM_MATCH_H_

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <signal.h>
#include <time.h>
#include<vector>


#ifndef ATTR_ALIGN
#  if defined(__GNUC__)
#    define ATTR_ALIGN(n)	__attribute__((aligned(n)))
#  else
#    define ATTR_ALIGN(n)	__declspec(align(n))
#  endif
#endif // #ifndef ATTR_ALIGN

using namespace cv;
using namespace std;

namespace shapematch {

	//ƥ��ģ��ķ�Χ
	struct MatchRange
	{
		float begin;
		float end;
		float step;

		MatchRange() : begin(0.f), end(0.f), step(0.f) {}
		MatchRange(float b, float e, float s);
	};
	inline MatchRange::MatchRange(float b, float e, float s) : begin(b), end(e), step(s) {}
	typedef struct MatchRange AngleRange;
	typedef struct MatchRange ScaleRange;

	//����ģ����״��
	typedef struct ShapeInfo_S
	{
		float angle;
		float scale;
	}ShapeInfo;

	//������������
	typedef struct Feature_S
	{
		int x;
		int y;
		int lbl;			//Ϊ������ķ���
	}Feature;

	//��ѡ���ɸѡ��ѡ��ǰ�����÷ָߵĵ�
	typedef struct Candidate_S
	{
		//�ѵ÷ָߵĺ�ѡ������ǰ��
		bool operator<(const struct Candidate_S &rhs) const
		{
			return score > rhs.score;
		}
		float score;
		Feature feature;

	}Candidate;

	//����ģ��ṹ��
	typedef struct Template_S
	{
		int id = 0;
		int pyramid_level = 0;
		int is_valid = 0;
		int x = 0;
		int y = 0;
		int w = 0;
		int h = 0;
		cv::Point relaCenter = cv::Point(0, 0);
		ShapeInfo shape_info;
		vector<Feature> features;
		int InitEdgeNum;

	}Template;

	//����ƥ����
	typedef struct Match_S
	{
		//��������ƶȵ�������ǰ��
		bool operator<(const struct Match_S &rhs) const
		{
			//Ϊ���������ƶ��ظ����ڶ����������ģ��id
			if (similarity != rhs.similarity)
				return similarity > rhs.similarity;
			else
				return template_id < rhs.template_id;
		}

		bool operator==(const struct Match_S &rhs) const
		{
			return x == rhs.x && y == rhs.y && similarity == rhs.similarity;
		}

		int x;
		int y;
		float similarity;
		int template_id;
		int feature_size;
		int RightNum = 0;
	}Match;

	typedef enum PyramidLevel_E
	{
		PyramidLevel_0 = 0,
		PyramidLevel_1 = 1,
		PyramidLevel_2 = 2,
		PyramidLevel_3 = 3,
		PyramidLevel_4 = 4,
		PyramidLevel_5 = 5,
		PyramidLevel_6 = 6,
		PyramidLevel_7 = 7,
		PyramidLevel_TabooUse = 16,
	}PyramidLevel;

	typedef enum MatchingStrategy_E
	{
		Strategy_Accurate = 0,
		Strategy_Middling = 1,
		Strategy_Rough = 2,
	}MatchingStrategy;


	typedef struct PointCluster {
		float x = 0.0;
		float y = 0.0;
		float mag = 0.0;
		float ang = 0.0;
	}PT;

	//��ȡHSVͨ����
	typedef struct HSV_info {
		cv::Mat ChannelImage;
		int channel;
	}HSV_I;
	//���ھ���
	class KCluster {
	public:
		void doCluster(cv::Mat &model, cv::Mat mag, cv::Mat angle, std::vector<cv::Point2f>vPt, int k);
	protected:
		//��ȡ����
		float getDistance(PT point, PT center);

	private:
		//�㼯��Ŀ
		int POINTNUM = 1000;
		// ������
		int k = 3;
		//��������
		int ITER = 100;
	};

	class ShapeMatching
	{
	public:

		ShapeMatching(string model_root, string class_name);
		~ShapeMatching();
		/*
		@model: ����ͼ��
		@angle_range: �Ƕȷ�Χ
		@scale_range: �߶ȷ�Χ
		@num_features: ������
		@weak_thresh������ֵ
		@strong_thresh: ǿ��ֵ
		@mask: ����
		*/

		//1.����ģ��
		void MakingTemplates(Mat &model, AngleRange angle_range, ScaleRange scale_range,
			int num_features, float weak_thresh = 30.0f, float strong_thresh = 60.0f,
			Mat mask = Mat());
		/*
		2.����ģ��
		*/
		void LoadModel();
		/*
		@source: ����ͼ��
		@score_thresh: ƥ�������ֵ
		@overlap: �ص���ֵ
		@mag_thresh: ��С�ݶ���ֵ
		@greediness: ̰���ȣ�ԽСƥ��Խ�죬���ǿ����޷�ƥ�䵽Ŀ��
		@pyrd_level: ������������Խ��ƥ��Խ�죬���ǿ����޷�ƥ�䵽Ŀ��
		@T: T����
		@top_k: ���ƥ����ٸ�
		@strategy: ��ȷƥ��(0), ��ͨƥ��(1), ����ƥ��(2)
		@mask: ƥ������
		*/

		//3.��ʼƥ��
		vector<Match> Matching(Mat source, float score_thresh = 0.9f, float overlap = 0.4f,
			float mag_thresh = 30.f, float greediness = 0.9f, PyramidLevel pyrd_level = PyramidLevel_2,
			int T = 2, int top_k = 0, MatchingStrategy strategy = Strategy_Accurate, const Mat mask = Mat());

		//4.����ͼ��
		void DrawMatches(Mat &image, vector<Match> matches, Scalar color);

	protected:
		void PaddingModelAndMask(Mat &corner,Mat &model, Mat &mask, float max_scale);
		vector<ShapeInfo> ProduceShapeInfos(AngleRange angle_range, ScaleRange scale_range);
		Mat Transform(Mat src, float angle, float scale);
		Mat MdlOf(Mat model, ShapeInfo info);
		Mat MskOf(Mat mask, ShapeInfo info);
		void DrawTemplate(Mat &image, Template templ, Scalar color);
		void QuantifyEdge(Mat image, Mat &angle, Mat &quantized_angle, Mat &mag, float mag_thresh, bool calc_180 = true);
		void Quantify8(Mat angle, Mat &quantized_angle, Mat mag, float mag_thresh);
		void Quantify180(Mat angle, Mat &quantized_angle, Mat mag, float mag_thresh);
		Template ExtractTemplate(Mat model,Mat angle, Mat quantized_angle, Mat mag, ShapeInfo shape_info,
			PyramidLevel pl, float weak_thresh, float strong_thresh, int num_features, Mat mask);
		Template SelectScatteredFeatures(vector<Candidate> candidates, int num_features, float distance);
		Rect CropTemplate(Template &templ, cv::Mat angle);
		void LoadRegion8Idxes();
		void ClearModel();
		void SaveModel();
		void InitMatchParameter(float score_thresh, float overlap, float mag_thresh, float greediness, int T, int top_k, MatchingStrategy strategy);
		void GetAllPyramidLevelValidSource(Mat &source, PyramidLevel pyrd_level);
		vector<Match> GetTopKMatches(vector<Match> matches);
		vector<Match> DoNmsMatches(vector<Match> matches, PyramidLevel pl, float overlap);
		vector<Match> MatchingPyrd180(Mat src, PyramidLevel pl, vector<int> region_idxes = vector<int>());
		vector<Match> MatchingPyrd8(Mat src, PyramidLevel pl, bool isTopLevel, vector<int> region_idxes = vector<int>());
		void Spread(const Mat quantized_angle, Mat &spread_angle, int T);
		void ComputeResponseMaps(const Mat spread_angle, vector<Mat> &response_maps);
		bool CalcPyUpRoiAndStartPoint(PyramidLevel cur_pl, PyramidLevel obj_pl, Match match,
			Mat &r, Point &p, bool is_padding = false);
		void CalcRegionIndexes(vector<int> &region_idxes, Match match, MatchingStrategy strategy);
		vector<Match> ReconfirmMatches(vector<Match> matches, PyramidLevel pl);
		vector<Match> MatchingFinal(vector<Match> matches, PyramidLevel pl);

		void girdLength(cv::Mat model, int pl);
		//����ɸѡ������
		std::vector<Candidate> gridSelectFeaturePoints(cv::Mat grayImg, 
			cv::Mat angle, cv::Mat quantized_angle, cv::Mat mag, std::vector<Candidate> candidates);
		//˫��ֵ������ȡĿ��ͼ���Ե��canny����
		cv::Mat SourceExtractTemplate(Mat angle, Mat quantized_angle, Mat mag,
			float weak_thresh, float strong_thresh, Mat mask = Mat());

		//���ٻ���ͼ�ļ���
		cv::Mat Integral_row(cv::Mat src);

		//��ȡ���ʺ�ƥ���ͨ��
		cv::Mat rightChannel(cv::Mat &image);

		//��ȡ���ʵ�HSVͨ��
		HSV_I getHSVinfo(cv::Mat &imageRGB, int channelIndex, bool flag);

		//����ģ��ͼ���mask
		cv::Mat makeMask(cv::Mat &templ);

		//����ͼ������ֵ�͵ĺ���
		double sumPixel(cv::Mat &img);

		//��ȡ��Դͼ���HSVͨ��ͼ��
		cv::Mat sourceHSVChannel(cv::Mat &source, int index);

		//ɸѡ��Դͼ������򣬼ӿ�ƥ���ٶ�
		cv::Mat filterSource(cv::Mat &source);

		//����ǵ���Ӧͼ�������д���
		cv::Mat calcuCorner(cv::Mat &image);

	private:
		typedef vector<Template> TemplateMatchRange;
		TemplateMatchRange templ_all_[PyramidLevel_TabooUse];
		vector<Mat> sources_;
		ATTR_ALIGN(32) float score_table_[180][180];
		ATTR_ALIGN(8) unsigned char score_table_8map_[8][256];
		string model_root_;
		string class_name_;
		AngleRange angle_range_;
		ScaleRange scale_range_;
		vector<int> region8_idxes_;

		float score_thresh_;
		float overlap_;
		float mag_thresh_;
		float greediness_;
		int T_;
		int top_k_;
		MatchingStrategy strategy_;

		//���ñ�Ե��ȡ����ֵ
		float weak_thresh_;
		float strong_thresh_;

		int grid_length_;
		cv::Mat model_;

		HSV_I modelHSV_;

		int padding_;
		int min_x_;
		int min_y_;
	};
}

void  getFilePaths(vector<cv::String>& filepaths, cv::String filePath); //��sort����

		//��ȡ���ʺ�ƥ���ͨ��
int rightChannelNum(cv::Mat &image);
//��ȡͨ��ͼ��
cv::Mat sourceChannel(cv::Mat &source, int channel);

//��ȡ��Դͼ���HSVͨ��ͼ��
cv::Mat sourceHSVChannel(cv::Mat &source, int index);



#endif

