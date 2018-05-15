#include "SignedRandomProjection.h"
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;
#pragma once

SignedRandomProjection::SignedRandomProjection(int dimention, int numOfHashes, int ratio) {
    _dim = dimention;
    _numhashes = numOfHashes;
    _samSize = ceil(1.0*_dim / ratio);
    cout<<_samSize<<endl;

    int *a = new int[_dim];
	// a[0]=0;
    for (size_t i = 0; i < _dim; i++) {
        a[i] = i;
    }

    srand(time(0));
    _randBits = new short *[_numhashes];
    _indices = new int *[_numhashes];
//#pragma omp parallel for
    for (size_t i = 0; i < _numhashes; i++) {
        random_shuffle(&a[0], &a[_dim - 1]);
        _randBits[i] = new short[_samSize];
        _indices[i] = new int[_samSize];

        for (size_t j = 0; j < _samSize; j++) {
            _indices[i][j] = a[j];
		// _indices[i][0] = _dim-1;
            int curr = rand();
            if (curr % 2 == 0) {
                _randBits[i][j] = 1;
            } else {
                _randBits[i][j] = -1;
            }
        }
    }
}

int *SignedRandomProjection::getHash(double *vector, int length) {
    // length should be = to _dim
    int *hashes = new int[_numhashes];

 // #pragma omp parallel for
    for (int i = 0; i < _numhashes; i++) {
        double s = 0;
        for (size_t j = 0; j < _samSize; j++) {
            double v = vector[_indices[i][j]];
            if (_randBits[i][j] >= 0) {
                s += v;
            } else {
                s -= v;
            }
        }
        hashes[i] = (s >= 0 ? 0 : 1);
        //cout << hashes[i] << endl;
//        printf("s = %f, hash[%ld] = %d\n", s, i, hashes[i]);
    }
    return hashes;
}


int *SignedRandomProjection::getHashForTables(double *vector, int K, int tableid, int p_or_n) {
    // length should be = to _dim
    int *hashes = new int[K];
// #pragma omp parallel for
    for (int i = tableid*K; i < (tableid+1)*K; i++) {
        double s = 0;
        for (size_t j = 0; j < _samSize; j++) {
            double v = vector[_indices[i][j]];
            if (_randBits[i][j] >= 0) {
                s += v;
            } else {
                s -= v;
            }
        }
        hashes[i-tableid*K] = (s >= 0 ? 0 : 1);
    }
    return hashes;
}


SignedRandomProjection::~SignedRandomProjection() {
    for (size_t i = 0; i < _numhashes; i++) {
        delete[]   _randBits[i];
        delete[]   _indices[i];
    }
    delete[]   _randBits;
    delete[]   _indices;
}
