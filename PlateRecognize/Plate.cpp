#include "Plate.h"


string Plate::str()
{
	string result = "";
	//�����Ҿ��û����������λ�ú��ַ����������ٽ���sort����
	vector<int> orderIndex;
	vector<int> xpositions;

	for ( int i = 0; i < charsPos.size(); ++i )
	{
		orderIndex.push_back(i);
		xpositions.push_back(charsPos[i].x);
	}
	float min=xpositions[0];
    int minIndex=0;
	for ( int i = 0; i < xpositions.size(); ++i )
	{
		float min = xpositions[i];
		int minIndex = i;
		for ( int j = i; j < xpositions.size(); ++j )
		{
			if ( xpositions[j] < min )
			{
				min = xpositions[j];
				minIndex = j;
			}
		}

		int aux_i = orderIndex[i];
		int aux_min = orderIndex[minIndex];
		orderIndex[i] = aux_min;
		orderIndex[minIndex] = aux_i;

		float aux_xi = xpositions[i];
		float aux_xmin = xpositions[minIndex];
		xpositions[i] = aux_xmin;
		xpositions[minIndex] = aux_xi;

	}
	for ( int i = 0; i < orderIndex.size(); ++i )
		result += chars[ orderIndex[i] ]; 



	return result;
}