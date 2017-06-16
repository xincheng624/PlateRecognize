#include "OCR.h"
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>


using namespace cv;

const int OCR::numCharacters = 30;
const char OCR::strCharacters[] = {'0','1','2','3','4','5','6','7','8','9','B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'};

OCR::OCR( string filename )
{
	trained = false;
	saveSegment = false;
	charSize = 20;

	FileStorage fs;
	fs.open( filename, FileStorage::READ );
	Mat TrainingData;
	Mat Classes;
	fs["TrainingDataF15"] >> TrainingData;
	fs["classes"] >> Classes;
	train( TrainingData, Classes, 10 );
}

void OCR::run( Plate *input )
{
	vector<CharSegment> segments = segment( *input );
	for ( int i = 0; i < segments.size(); ++i )
	{
		Mat ch = segments[i].img;

		Mat f = feature( ch, 15 );//sizeData 15X15
		int character = classify( f );

		input->chars.push_back( strCharacters[ character] );
		input->charsPos.push_back ( segments[i].rect );

	}
}

vector<CharSegment> OCR::segment( Plate plate )
{
	Mat input = plate.plateImg;
	vector<CharSegment> output;

	Mat img_threshold;
	threshold( input, img_threshold,
		60,//阈值，涉及到二值化后字符的清晰度，55也是个不错的选项
		255, CV_THRESH_BINARY_INV );//反转是因为findContour查找白色像素

	Mat img_contours;
	img_threshold.copyTo( img_contours );
	vector< vector<Point> > contours;
	findContours( img_contours, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE ); 
	
	//drawContours( img_threshold, contours, -1, Scalar(255,0,0),1);
	//imshow("1",img_threshold);
	//waitKey();
	for( vector< vector<Point> >::iterator it = contours.begin();
		 it != contours.end(); ++it )
	{
		Rect mr = boundingRect( Mat(*it) );
		Mat auxROI( img_threshold, mr);//获取感兴趣区域
		if ( verifySize( auxROI ) )
		{
			auxROI = processChar( auxROI );
			output.push_back( CharSegment( auxROI, mr ) );
		}
		//imshow("1", *it );
		//waitKey();
	}
	

	return output;
}

bool OCR::verifySize( Mat r)
{
	float aspect = 45.0f/77.0f;
	float charAspect = (float)r.cols/(float)r.rows;
	float error = 0.35;
	float minHeight = 15;
	float maxHeight = 28;

	float minAspect = 0.2;//由于字符1的宽高比较小，取0.2
	float maxAspect = aspect + aspect*error;

	float area = static_cast<float>( countNonZero(r) );
	float rArea = static_cast<float>( r.cols*r.rows );

	float percPixels = area/rArea;//字符占区域的比例，小于0.8视为黑块
	if ( percPixels < 0.8 && charAspect > minAspect && charAspect < maxAspect 
		&& r.rows >= minHeight && r.rows <= maxHeight )
		return true;
	else
		return false;

}

Mat OCR::ProjectedHistogram( Mat img, int t )
{
	int sz = (t)? img.rows : img.cols;
	Mat mhist = Mat::zeros( 1, sz, CV_32F);

	for (int j = 0; j < sz; ++j )
	{
		Mat data = (t)? img.row(j):img.col(j);
		mhist.at<float>(j) = static_cast<float>( countNonZero( data ) );
	}

	double min, max;
	minMaxLoc( mhist, &min, &max );

	if ( max > 0 )
		mhist.convertTo( mhist, -1, 1.0f/max , 0 );

	return mhist;
}

Mat OCR::feature( Mat in, int sizeData )
{
	Mat vhist = ProjectedHistogram( in, VERTICAL );
	Mat hhist = ProjectedHistogram( in, HORIZONTAL );
	Mat lowData;
	resize( in, lowData, Size( sizeData, sizeData ) );
	int numcols = vhist.cols + hhist.cols + lowData.cols*lowData.cols;

	Mat out = Mat::zeros( 1, numcols, CV_32F );
	int j = 0;
	for ( int i = 0; i < vhist.cols; ++i )
	{
		out.at<float>(j) = vhist.at<float>(i);
		++j;
	}
	for ( int i = 0; i < hhist.cols; ++i )
	{
		out.at<float>(j) = hhist.at<float>(i);
		++j;
	}
	for ( int i = 0; i < lowData.cols; ++i )
	{
		for ( int k = 0; k < lowData.cols; ++k )
		{
			out.at<float>(j) = (float)lowData.at<unsigned char>(i,k);//为什么必须是unsigned char
			++j;
		}
	}
	return out;
}

void OCR::train( Mat TrainData, Mat classes, int nlayers )
{
	Mat layerSizes( 1, 3, CV_32SC1 );
	layerSizes.at<int>(0) = TrainData.cols;
	layerSizes.at<int>(1) = nlayers;
	layerSizes.at<int>(2) = numCharacters;
	ann.create( layerSizes, CvANN_MLP::SIGMOID_SYM, 1, 1 );

	Mat trainClasses;
	trainClasses.create( TrainData.rows, numCharacters, CV_32FC1 );
	for ( int i = 0; i < trainClasses.rows; ++i )
	{
		for ( int k = 0; k < trainClasses.cols; ++k )
		{
			if ( k == classes.at<int>(i) )
				trainClasses.at<float>(i,k) = 1;
			else
				trainClasses.at<float>(i,k) = 0;
		}
	}

	Mat weights( 1, TrainData.rows, CV_32FC1, Scalar::all(1) );
	ann.train( TrainData, trainClasses, weights);
	trained = true;
}

int OCR::classify( Mat f )
{
	int result = -1;
	Mat output( 1, numCharacters, CV_32FC1 );
	ann.predict( f, output );
	Point maxLoc;
	double maxVal;
	minMaxLoc( output, 0, &maxVal, 0, &maxLoc );

	return maxLoc.x;
}

Mat OCR::processChar( Mat input )
{
	int w = input.cols;
	int h = input.rows;
	Mat transformMat = Mat::eye(2,3,CV_32F);
	int m = max(w,h);
	transformMat.at<float>(0,2) = static_cast<float>( m/2 - w/2 );
	transformMat.at<float>(1,2) = static_cast<float>( m/2 - h/2 );

	Mat img_wrap(m,m,input.type());
	warpAffine( input, img_wrap, transformMat, img_wrap.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0) );

	resize( img_wrap, img_wrap, Size( charSize, charSize ) );//对图像形状大小处理为一致，以用于识别
	return img_wrap;
}