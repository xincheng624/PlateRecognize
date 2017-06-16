#ifndef PLATE_H
#define PLATE_H

#include <opencv2\core\core.hpp>
using namespace cv;
#include <string>
using std::string;

class Plate
{
public:
	Mat plateImg;
	Rect boundingR;

	Plate( Mat p, Rect r):plateImg(p),boundingR(r){}
	string str();

	vector<char> chars; //车牌号存储
	vector<Rect> charsPos;//车牌号的位置存储，每个字符对应一个矩形的位置
};

#endif