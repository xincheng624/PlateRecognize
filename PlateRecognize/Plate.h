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

	vector<char> chars; //���ƺŴ洢
	vector<Rect> charsPos;//���ƺŵ�λ�ô洢��ÿ���ַ���Ӧһ�����ε�λ��
};

#endif