#include <vector>
#pragma once
using namespace std;

class SignedRandomProjection 
{
private:
	int _dim;
	int _numhashes, _samSize;
	short ** _randBits;
	int ** _indices;
public:
	SignedRandomProjection(int dimention, int numOfHashes, int ratio);
	int * getHashForTables(double *vector, int K, int tableid, int p_or_n);
	int * getHash(double * vector, int length);
	~SignedRandomProjection();
};
