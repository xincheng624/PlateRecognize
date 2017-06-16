#ifndef OCR_H
#define OCR_H

#include <opencv2\core\core.hpp>
#include <opencv2\ml\ml.hpp>
#include <string>
#include "Plate.h"

using std::string;
using cv::Mat;
using cv::Rect;

const int VERTICAL = 0;
const int HORIZONTAL = 1;

//const int numCharacters = 30;


class CharSegment
{
public:
	CharSegment(){}
	CharSegment( Mat i, Rect r ):img(i),rect(r){} 
	Mat img;
	Rect rect;
};


class OCR
{
public:
	OCR(){};
	OCR( string filename );
	void run( Plate *input );
	vector<CharSegment> segment( Plate plate );
	bool verifySize( Mat r );
	Mat ProjectedHistogram( Mat img, int t );
	Mat feature( Mat in, int sizeData );
	void train( Mat TrainData, Mat classes, int nlayers );
	int classify( Mat f );
	Mat processChar( Mat input );

	static const int numCharacters;
	static const char strCharacters[];
	bool saveSegment;
private:
	CvANN_MLP ann;
	bool trained;
	
	int charSize;
};



#endif