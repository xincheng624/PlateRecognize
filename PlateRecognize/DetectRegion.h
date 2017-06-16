#ifndef DETECTREGION_H
#define DETECTREGION_H

//#include <opencv2\core\core.hpp>
#include <vector>
#include "Plate.h"
using cv::Mat;
using cv::RotatedRect;
using std::vector;

class DetectRegion
{

public:
	vector<Plate> segment( Mat input );
	bool verifySize( RotatedRect rect );
	//vector<Plate> output;
};

#endif