#include <iostream>
#include "Bucket.h"
#include <random>
#include <algorithm>
#include <vector>
#pragma once
using namespace std;
Bucket::Bucket()
{
	isInit = -1;
	totalAdded = 0;
}

Bucket::~Bucket()
{
	delete[] arr;
}
int Bucket::getSize()
{
	return _size;
}

void Bucket::shuffleData(int* index ,int size)
{
	// Choose random sample, using Mikolov's fast almost-uniform random number
	// _randomNumber = _randomNumber * (unsigned long long) 25214903917 + 11;
	// int randomIndex = _randomNumber % (unsigned long long) size;
	// cout<<randomIndex<<endl;	
	// return randomIndex;

    srand ( time(NULL) );
 
    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (int i = size-1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        int j = rand() % (i+1);
        // Swap arr[i] with the element at random index
        swap(index[i], index[j]);
    }
}

int Bucket::add(int id)
{
	totalAdded++;
	
	if (isInit == -1) {
		arr = new int[_size];
		for (int i = 0; i < _size; i++)
			arr[i] = 0;
		isInit = +1;
	}
	if (index == _size) {
		int currSamp = rand() % totalAdded;
		if (currSamp == (totalAdded-1));
		{
			int randind = rand() % _size;
			arr[randind] = id;
		}
	}
	else {
		arr[index] = id;
		index++;
	}
	return 1;
}

int Bucket::retrieve(int index)
{
	if (index >= _size)
		return -1;
	return arr[index];
}

int * Bucket::sample()
{
	int * sample = new int[2];
	if (index == 0)
	{
		//no sample found
		sample[0] = -1; 
		sample[1] = -1; 
	}
	
	int randint = rand() % index;
	sample[0] = arr[randint]; // return sample
	sample[1] = totalAdded; // probability of selecting = 1/totalAdded;
	return sample;
}

int * Bucket::sampleBatch(int size)
{
	int * sample;
	if (index == 0)
	{
		//no sample found
		sample[0] = -1; 
		sample[1] = -1; 
	}
	
	int randint = rand() % index;

	if (size>totalAdded){
		sample = new int[totalAdded+2];
		sample[0] = totalAdded;
		for (int i=0; i<totalAdded ;i++)
		{
			sample[2+i] = arr[i];
		}

	}else{
		sample = new int[size+2];
		sample[0] = size;
		// int *v = new int[totalAdded];
		// for (int i=0; i<totalAdded ;i++)
		// {
		// 	v[i] = i;
		// }
		// shuffleData(v, totalAdded);

		for (int i=0; i<size ;i++)
		{
			int randint = rand() % index;
			sample[2+i] = arr[randint];
		}

	}
	sample[1] = totalAdded; // probability of selecting = 1/totalAdded;
	return sample;
}

int * Bucket::getAll()
{
	if (isInit == -1)
		return NULL;
	return arr;
}