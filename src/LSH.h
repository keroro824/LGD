#pragma once
#include "Bucket.h"
#include "SignedRandomProjection.h"
#include <vector>
class LSH {
private:
	Bucket **_bucket;
	int _K;
	int _L;
	int *rand1;
	std::vector<int> _v;
public:
	LSH(int K, int L);
	int static _rangePow;
	int static _thresh;
	void add(int *hashes, int id);
	int * retrieve(int *hashes);
	int * sample(int *hashes);
	int * sample(double *query, SignedRandomProjection *srp, int p_or_n);
	void sampleBatch(double *query, SignedRandomProjection *srp, int batch, int **sample, int p_or_n);
	void count_unique();
	~LSH();
}; 