#include <iostream>
#include <unordered_map>
#include "HashFunction.h"
#include "SignedRandomProjection.h"
//#include "DensifiedMinHash.h"
#include "LSH.h"
//#include <ppl.h>
#include <random>
#include <algorithm>
#include <vector>
#include <climits>
#include <chrono>
#include <set>
#pragma once
/* Author: Anshumali Shrivastava
*  COPYRIGHT PROTECTION
*  Free for research use. 
*  For commercial use, contact:  RICE UNIVERSITY INVENTION & PATENT or the Author.
*/

using namespace std;
//using namespace concurrency;


LSH::LSH(int K, int L)
{
	_K = K;
	_L = L;
	//_range = 1 << 22;
	_bucket = new Bucket*[L];
	// cout<<"defined"<<_rangePow<<endl;
//#pragma omp parallel for
	for (int i = 0; i < L; i++)
	{
		_bucket[i] = new Bucket[1<<_rangePow];
	}

	rand1 = new int[_K*_L];
//	cout<<"success"<<endl;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, INT_MAX);
//	cout<<dis(gen)<<endl;
//#pragma omp parallel for
	for (int i = 0; i < _K*_L; i++)
	{
//		cout<<UINT_MAX<<endl;
		rand1[i] = dis(gen);
		//	cout<<"h"<<endl;
		if (rand1[i] % 2 == 0)	
			rand1[i]++;
//	cout<<"wd"<<endl;	
}

	std::vector<int> v(_L);
	for (int i = 0; i < _L; i++)
	{
		v[i] = i;
	}
	_v = v;

}

void LSH::add(int *hashes, int id)
{
	// #pragma opm parallel for
	for (int i = 0; i < _L; i++)
	{
		unsigned int index = 0;
		for (int j = 0; j < _K; j++)
		{
			unsigned int h = hashes[_K*i + j];
			// h *= rand1[_K*i + j];
			// h ^= h >> 13;
			// h ^= rand1[_K*i + j];
			// index += h*hashes[_K*i + j];
			index += h<<(_K-1-j);
		}
		// index = (index << 11) >> (32 - LSH::_rangePow);
		// index = index % (2<<(LSH::_rangePow-1));
		// index = (index% (2<<(LSH::_rangePow)));
		// cout<<"Table: "<<i<<" Query index "<< index<<endl;
		// cout <<i<<index<<endl;
		_bucket[i][index].add(id);
	}
}

void LSH::count_unique()
{
	std::set<int> total;
	for (int i = 0; i < _L; i++)
	{
		for (int j = 0; j < (1<<(_K)); j++)
		{
			int *arr = _bucket[i][j].getAll();
			// const size_t len = sizeof(arr) / sizeof(arr[0]);
			std::set<int> s(arr, arr + _bucket[i][j]._size);
			total.insert(s.begin(), s.end());
			std::cout <<  _bucket[i][j].totalAdded <<" ";
		}
		std::cout <<" "<<endl;
}
std::cout << total.size() << std::endl;



}



/*
returns an array with  ret with 3 values
ret[0] is the sample returned 
Sampling Probability = (1 - (1 - p^K)^ret[2])*(1/ret[1])
where p = (1 - 1/(Range))LSHCollprob(q,ret[0]) + 1/(Range)
Range = the range of hashtable which in our case is (1<<_rangePow)
*/

int * LSH::sample(int *hashes)
{

	int * samplewithProb = new int[3];
	samplewithProb[0] = -1;
	samplewithProb[1] = -1;
	samplewithProb[2] = -1;
	


	// std::random_device rd;
	// std::mt19937 g(rd());

	// std::shuffle(_v.begin(), _v.end(), g);


	for (int i = 0; i < _L; i++)
	{  
		// int table = _v[i];
		int table = rand()%_L;
		unsigned int index = 0;
		for (int j = 0; j < _K; j++)
		{
			unsigned int h = hashes[_K*table + j];
			// h *= rand1[_K*table + j];
			// h ^= h >> 13;
			// h ^= rand1[_K*table + j];
			// index += h*hashes[_K*table + j];
			index += h<<(_K-1-j);
		}
		// index = (index << 11) >> (32 - LSH::_rangePow);
		// cout<<"Table: "<<table<<" Query index "<< index<<endl;
		if (_bucket[table][index].getAll() == NULL)
		{
			continue;
		}
		else{
			int * retSamp = _bucket[table][index].sample();
			// cout<<"1"<<retSamp[0] << "2"<<retSamp[1] << "3"<<retSamp[0] <<endl;
			samplewithProb[0] = retSamp[0];
			samplewithProb[1] = retSamp[1];
			samplewithProb[2] = i+1;
			return samplewithProb;
		}
		
	}

	return samplewithProb;

}


int * LSH::sample(double *query, SignedRandomProjection *srp, int p_or_n)
{

	int * samplewithProb = new int[4];
	samplewithProb[0] = -1;
	samplewithProb[1] = -1;
	samplewithProb[2] = -1;
	samplewithProb[3] = -1;
	
// auto start_sam = std::chrono::steady_clock::now();

// 	std::random_device rd;
// 	std::mt19937 g(rd());

// 	std::shuffle(_v.begin(), _v.end(), g);
		// 		auto end_sam = std::chrono::steady_clock::now();
		// auto elapsed_sam = std::chrono::duration_cast<std::chrono::microseconds>(end_sam - start_sam);
  //       cout << "Iteration: "<< 0 <<" epoch took: " << elapsed_sam.count() << " microseconds."  <<std::endl;

	for (int i = 0; i < _L; i++)
	{  
		int table = rand()%_L;
		unsigned int index = 0;
		
		int * hashes = srp->getHashForTables(query, _K, table, p_or_n);

		for (int j = 0; j < _K; j++)
		{
			unsigned int h = hashes[j];
			// h *= rand1[_K*table + j];
			// h ^= h >> 13;
			// h ^= rand1[_K*table + j];
			// index += h*hashes[j];
			index += h<<(_K-1-j);
		}
		// index = (index << 11) >> (32 - LSH::_rangePow);
		// index = index % (2<<(LSH::_rangePow-1));

		if (_bucket[table][index].getAll() == NULL)
		{
			continue;
		}
		else{
			
			int * retSamp = _bucket[table][index].sample();


			// cout<<"Table: "<<table<<" Query index "<< index<<endl;
			// cout<<"1"<<retSamp[0] << "2"<<retSamp[1] << "3"<<retSamp[0] <<endl;
			samplewithProb[0] = retSamp[0];
			samplewithProb[1] = retSamp[1];
			samplewithProb[2] = i+1;
			samplewithProb[3] = index;
			return samplewithProb;
		}
	}
	return samplewithProb;

}


void LSH::sampleBatch(double *query, SignedRandomProjection *srp, int batch, int **samplewithProb, int p_or_n)
{


	// cout<<samplewithProb[0][0]<<endl;
	int sampled = 0;

	// std::random_device rd;
	// std::mt19937 g(rd());

	// std::shuffle(_v.begin(), _v.end(), g);

	for (int i = 0; i < _L; i++)
	{  
		int table = rand()%_L;

		unsigned int index = 0;
		int * hashes = srp->getHashForTables(query, _K, table, p_or_n);
		for (int j = 0; j < _K; j++)
		{
			unsigned int h = hashes[j];
			// h *= rand1[_K*table + j];
			// h ^= h >> 13;
			// h ^= rand1[_K*table + j];
			// index += h*hashes[j];
			// cout << h <<endl;
			index += h<<(_K-1-j);
		}
		// index = (index << 11) >> (32 - LSH::_rangePow);
		// index = index % (2<<(LSH::_rangePow-1));
		// cout<<"Table: "<<table<<" Query index "<< index<<endl;
		if (_bucket[table][index].getAll() == NULL)
		{
			continue;
		}
		else{
			int * retSamp = _bucket[table][index].sampleBatch(batch-sampled);
			// cout<<"1"<<retSamp[0] << "2"<<retSamp[1] << "3"<<retSamp[0] <<endl;
			// samplewithProb[0] = retSamp[0];
			// samplewithProb[1] = retSamp[1];
			// samplewithProb[2] = i+1;
			for (int s=0; s< retSamp[0] ;s++)
			{
				samplewithProb[0][sampled+s] = retSamp[s+2];
				samplewithProb[1][sampled+s] = retSamp[1];
				samplewithProb[2][sampled+s] = i+1;
			}

			sampled+= retSamp[0];
			if (sampled == batch ){
				// cout<<"sampled "<<samplewithProb[0][0]<<endl;
				// return samplewithProb;
				return;
			}
			
		}
		
	}

	// return samplewithProb;

}


LSH::~LSH()
{
	
	for (size_t i = 0; i < _L; i++)
	{
		delete[] _bucket[i];
	}
	delete[] _bucket;
}
