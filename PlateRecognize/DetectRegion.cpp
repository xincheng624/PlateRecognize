#include <opencv2\imgproc\imgproc.hpp>
//#include <opencv2\highgui\highgui.hpp>
#include "DetectRegion.h"
#include <time.h>
//#include "Plate.h"
using namespace cv;
using std::vector;
vector<Plate> DetectRegion::segment( Mat input )
{
	vector<Plate> output;

	Mat img_gray;
	cvtColor( input, img_gray, CV_BGR2GRAY );
	blur( img_gray, img_gray, Size(5,5) );

	Mat img_sobel;
	Sobel( img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0);

	Mat img_threshold;
	threshold( img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY );

	Mat element = getStructuringElement( MORPH_RECT, Size(17,3) );
	morphologyEx( img_threshold, img_threshold, CV_MOP_CLOSE, element);        //�ղ����ѳ��ƾ������������������������������״������Ĳ�����ȥ;

	vector< vector<Point> > contours;
	findContours( img_threshold, contours,
		CV_RETR_EXTERNAL, //ֻ��ȡ������
		CV_CHAIN_APPROX_NONE );//�����������ת��Ϊ��
	vector< vector<Point> >::iterator itc = contours.begin();
	vector< RotatedRect > rects;
	while ( itc != contours.end() )
	{
		RotatedRect mr = minAreaRect( Mat(*itc) );
		if ( !verifySize(mr) ) //�ҳ���״������ڸ����������������Ҫ��ľ���
		{
			itc = contours.erase(itc);
		}
		else
		{
			++itc;
			rects.push_back(mr);
		}
	}

	Mat result;
	input.copyTo(result);
	drawContours( result, contours, -1, Scalar(255,0,0), 8); //�������б߽�,����ķ������ǻ�����
	/*for ( vector<RotatedRect>::iterator it = rects.begin(); it != rects.end(); ++it )
	{
		Point2f ver[4];
		it->points( ver ); //��ȡRotatedRect���ĸ��ǵ�
		for ( int i = 0; i < 4; ++i )
		{
			line( img_gray, ver[i], ver[(i+1)%4], Scalar( 255, 0, 0),8); //�ĵ㻭�������
		}

	}*/

	for ( int i = 0; i < rects.size(); ++i)
	{
		circle( result, rects[i].center, 3, Scalar(255,0,0), -1 ); //����
		float minSize  = ( rects[i].size.width < rects[i].size.height ) ?
			rects[i].size.width : rects[i].size.height;
		minSize /= 2;

		srand( time(NULL) );
		Mat mask;
		mask.create( input.rows+2, input.cols+2, CV_8UC1 );
		mask = Scalar::all( 0 );

		int loDiff = 30;
		int upDiff = 30;
		int connectivity = 4;
		int newMaskVal = 255;
		int NumSeeds = 10;
		Rect ccomp;
		int flags = connectivity + ( newMaskVal << 8 ) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;

		for ( int j = 0; j < NumSeeds; ++j )
		{
			Point seed;
			seed.x = rects[i].center.x + rand()%(int)minSize - (minSize/2); //����x ���� minSize/2��Χ�������
			seed.y = rects[i].center.y + rand()%(int)minSize - (minSize/2);
			circle( result, seed, 1, Scalar(0,255,255), -1);
			int area = floodFill( input, mask, seed, Scalar(0,255,255), &ccomp,
				Scalar(loDiff,loDiff,loDiff), Scalar(upDiff,upDiff,upDiff), flags );
			//��rects�����������ˮ��䣬������ǳ��ƣ����������������ܴ���ߺ�С��ֻ�г��ƵĲ�λ�������ڳ��Ƶ���״
		}
		
		vector<Point> pointInterest;
		Mat_<uchar>::iterator itMask = mask.begin<uchar>();
		Mat_<uchar>::iterator end = mask.end<uchar>();
		for ( ; itMask != end; ++itMask )
		{
			if ( *itMask == 255 )
				pointInterest.push_back( itMask.pos() );
			
		}

		RotatedRect minRect = minAreaRect( pointInterest );
		if ( verifySize(minRect) )
		{
			Point2f rp[4];
			minRect.points( rp );
			for ( int k = 0; k < 4; ++k )
				line( result, rp[k], rp[(k+1)%4], Scalar(0,0,255),8);

			float r = (float)( minRect.size.width ) / (float)( minRect.size.height );
			float angle = minRect.angle;
			if ( r < 1 )
				angle += 90;  // r<1,˵�����ƾ��ο�С�ڸߣ�˵��ƫ���������ת90��;
			Mat rotmat = getRotationMatrix2D( minRect.center, angle, 1);

			Mat img_rotate;
			warpAffine( input, img_rotate, rotmat, input.size(), //���α任
				CV_INTER_CUBIC );//������ֵ
			Size rect_size = minRect.size;
			if ( r < 1 )
				std::swap( rect_size.width, rect_size.height );
			Mat img_crop;
			getRectSubPix( img_rotate, rect_size, minRect.center, img_crop );//���¸���Ȥ�ľ�������

			Mat resultResized;
			resultResized.create( 33, 144, CV_8UC3 );
			resize( img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC );//�����ƴ�Сͳһ��

			Mat grayResult;
			cvtColor( resultResized, grayResult, CV_BGR2GRAY );
			blur( grayResult, grayResult, Size(3,3) );
			equalizeHist( grayResult, grayResult );//ֱ��ͼ����

			
			output.push_back( Plate( grayResult, minRect.boundingRect() ) );
			//imshow("result",grayResult);
			//waitKey(0);
		}
		
	}

	return output;
}

bool DetectRegion::verifySize( RotatedRect rect )
{
	const float error = 0.4;
	const float aspect = 4.7272; //��߱�
	int minR = static_cast<int>( 15*aspect*15 ); 
	int maxR = static_cast<int>( 125*aspect*125 );
	float amin = aspect*( 1 - error );
	float amax = aspect*( 1 + error );
	int area = rect.size.height * rect.size.width;
	float rAspect = (float)( rect.size.width ) / (float)( rect.size.height );
	if ( rAspect < 1  && rAspect != 0 )
		rAspect = 1/rAspect;

	if ( ( area < minR || area > maxR ) || ( rAspect < amin || rAspect > amax ))
		return false;
	else
		return true;
}