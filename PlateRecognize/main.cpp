#include <iostream>
#include <sstream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>

using namespace cv;
using std::cout;
using std::endl;
using std::stringstream;

#include "DetectRegion.h"
#include "OCR.h"

int main ( int argc, char * argv[] )
{
	//string filename = "3732FWW.jpg";
	//Mat img = imread( "3028BYS.jpg" );
	//Mat img = imread( "3154FFY.jpg" );
	//Mat img = imread( "3266CNT.jpg" );
	//Mat img = imread( "3732FWW.jpg" );
	//Mat img = imread( "7215BGN.jpg" );

	for( int i = 1; i < argc; ++i)
	{
	Mat img = imread( argv[i] );
	DetectRegion detect;
	vector<Plate> possible_regions;
	possible_regions = detect.segment( img );


	//SVM���з���
	FileStorage fs;
	fs.open( "SVM.xml", FileStorage::READ );
	Mat SVM_TrainingData;
	Mat SVM_Classes;

	fs["TrainingData"] >> SVM_TrainingData;
	fs["classes"] >> SVM_Classes;

	CvSVMParams SVM_params;
	SVM_params.kernel_type = CvSVM::LINEAR;//SVM�������ã����������Ĳ�����������
	CvSVM svmClassifier( SVM_TrainingData, SVM_Classes, Mat(), Mat(), SVM_params );

	vector<Plate> plates;
	for ( int i = 0; i < possible_regions.size(); ++i )
	{
		Mat img = possible_regions[i].plateImg;
		Mat p = img.reshape(1,1);//ת��Ϊ1ͨ����1�е�����
		p.convertTo(p,CV_32FC1);//�������ͺ������Ϊ32����������ͨ��

		int response =  static_cast<int>( svmClassifier.predict(p) ); //���з���

		if ( response == 1 )
			plates.push_back( possible_regions[i] );

	}
	cout<<"��⵽��������"<< plates.size() << endl;

	OCR ocr("OCR.xml");//ע��xml��XML�ǲ�һ����
	ocr.saveSegment = true;
	vector<CharSegment> seg;
	/*for ( vector<Plate>::iterator it = plates.begin(); it != plates.end(); ++it )
	{		
		ocr.run( &*it );
		
	}*/

	for (int i = 0; i < plates.size(); ++i )
	{
		Plate plate = plates[i];
		ocr.run( &plate );
		string s = plate.str();
		cout<<"================================================"<<endl;
		cout<<"���ƺţ�  "<<s<<endl;
		cout<<"================================================"<<endl;
		rectangle( img, plate.boundingR, Scalar(0,0,255) );
		putText( img, s, Point( plate.boundingR.x, plate.boundingR.y ), CV_FONT_HERSHEY_SIMPLEX, 1,Scalar( 0, 255, 0 ), 2 );
	}
	imshow("img",img);
	waitKey();
	}
}