#include <iostream>
#include "Bucket.h"
#include <unordered_map>
#include "HashFunction.h"
#include "LSH.h"
#include "Sgd.h"
//#include <ppl.h>
#include <fstream>
#include <cstring>
#include <ctime>
#include <typeinfo>
#include <vector>
#include <chrono>
#pragma once
//#include <Windows.h>
//using namespace concurrency;
using namespace std;

/* Author: Beidi Chen
*  COPYRIGHT PROTECTION
*  Free for research use.
*  For commercial use, contact:  RICE UNIVERSITY INVENTION & PATENT or the Author.
*/

int Bucket::_size = 64;
int LSH::_rangePow = 24;
int K;
int L;
int ngrams = 3;
int chunk;
string outputFile;
string train_data_name;
string train_label_name;
string test_label_name;
string test_data_name;
string table_data_name;

int MinHashChunkSize = 32;

int width = 0;
int height = 0;

double **table_data;
double **train_data;
double **test_data;
double **train_label;
double **test_label;

int epoch = 10;
double conv = 0.00001;
double lr = 0.0001;
int trainNum = 5;
int testNum = 5;
double decayrate = 0.01;
double reg = 0.15;
int srpratio = 30;
int minibatch = 32;
int adagrad = 0;
int rangePow = 1;

// plain SGD: 0
// LSH SGD: 1
int type = 0;

string trim(string& str)
{
	size_t first = str.find_first_not_of(' ');
	size_t last = str.find_last_not_of(' ');
	return str.substr(first, (last - first + 1));
}


void parseconfig(string filename)
{
	string * arguments = new string[5];
	std::ifstream file(filename);
        if(!file)
        {
           cout<<"Error Config file not found: Given Filename "<< filename << endl;
         }
	std::string str;
	while (getline(file, str))
	{ 
		if (str == "")
			continue;

		std::size_t found = str.find("#");
		if (found != std::string::npos)
			continue;
		
		if (trim(str).length() < 3)
			continue;
		
		int index = str.find_first_of("=");
		string first = str.substr(0, index);
		string second = str.substr(index + 1, str.length());

		if (trim(first) == "K")
		{
			K = atoi(trim(second).c_str());
		}
		else if (trim(first) == "L")
		{
			L = atoi(trim(second).c_str());
		}
		else if (trim(first) == "shingles")
		{
			ngrams = atoi(trim(second).c_str());
		}
		else if (trim(first) == "TestNum")
		{
			testNum = atoi(trim(second).c_str());
		}
		else if (trim(first) == "TableData")
		{
			table_data_name = trim(second);
		}
		else if (trim(first) == "TrainData")
		{
			train_data_name = trim(second);
		}
		else if (trim(first) == "TestData")
		{
			test_data_name = trim(second);
		}
		else if (trim(first) == "TrainLabel")
		{
			train_label_name = trim(second);
		}
		else if (trim(first) == "TestLabel")
		{
			test_label_name = trim(second);
		}
		else if (trim(first) == "Output")
		{
			outputFile = trim(second);
		}
		else if (trim(first) == "BucketSize")
		{
			Bucket::_size = atoi(trim(second).c_str());
		}
		else if (trim(first) == "MinHashChunkSize")
		{
			MinHashChunkSize = atoi(trim(second).c_str());
		}
		else if (trim(first) == "Dim")
		{
			width = atoi(trim(second).c_str());
		}
		else if (trim(first) == "TrainNum")
		{
			trainNum = atoi(trim(second).c_str());
		}
		else if (trim(first) == "Convergence")
		{
			conv = atof(trim(second).c_str());
		}
		else if (trim(first) == "Lr")
		{
			lr = atof(trim(second).c_str());
		}
		else if (trim(first) == "Decay")
		{
			decayrate = atof(trim(second).c_str());
		}
		else if (trim(first) == "Epoch")
		{
			epoch = atoi(trim(second).c_str());
		}
		else if (trim(first) == "Reg")
		{
			reg = atof(trim(second).c_str());
		}
		else if (trim(first) == "SrpRatio")
		{
			srpratio = atoi(trim(second).c_str());
		}
		else if (trim(first) == "Minibatch")
		{
			minibatch = atoi(trim(second).c_str());
		}
		else if (trim(first) == "AdaGrad")
		{
			adagrad = atoi(trim(second).c_str());
		}

		else if (trim(first) == "RangePow")
		{
			rangePow = atoi(trim(second).c_str());
			LSH::_rangePow = rangePow;
			if (LSH::_rangePow >= 32)
				cout << "Range of Hash Values Too Big" << endl;
		}
		else if (trim(first) == "Type")
		{
			type = atoi(trim(second).c_str());
		}
		else
		{
			cout << "Error Parsing conf File at Line" << endl;
			cout << str << endl;
		}
	}
}

double** readData(string name, int number, int dim)
{
	if (name[name.size()-1]==13){
		name = name.substr(0, name.size()-1);
	}
    std::ifstream stream(name);
    if(!stream.is_open())
    {
       cout<<"Error inputcsvfile not found: Given Filename "<< name << endl;
		return 0;
    }  

    double** input = new double *[number];
    for( int i = 0; i < number; i++ ) {
        input[i] = new double[dim];
    }

 	double temp;
    for( int x = 0; x < number; x++ ) {
        for( int y = 0; y < dim; y++ ) {
        	stream >> temp;
            input[x][y] = temp;
        }
    }

    stream.close();
	return input;
}


void LshSgd(int argc,char *arg[])
{
	//Initialization
	parseconfig(arg[1]);

	// preprocess input data to Vector format (double)

	train_data = readData(train_data_name, trainNum, width);
	train_label = readData(train_label_name, trainNum, 1);
	test_data = readData(test_data_name, testNum, width);
	test_label = readData(test_label_name, testNum, 1);
	table_data = readData(table_data_name, trainNum, width+1);


	if (type==0){

		Sgd *_sgd = new Sgd(train_data, table_data, train_label, test_data, test_label, trainNum, testNum, width, conv, lr, epoch, decayrate, reg, 1, outputFile);
		_sgd->SGDUpdate(adagrad);

	}
	else if (type==3){

		Sgd *_sgd = new Sgd(train_data, table_data, train_label, test_data, test_label, trainNum, testNum, width, conv, lr, epoch, decayrate, reg, 1, outputFile);
		_sgd->GDUpdate(adagrad);

	}
	else if (type==1){


		// for (K=1;K<10; K++){

		LSH::_rangePow = K;
		if (LSH::_rangePow >= 32)
			cout << "Range of Hash Values Too Big" << endl;

		auto start = std::chrono::steady_clock::now();
		LSH *_Algo = new LSH(K, L);
	    auto end = std::chrono::steady_clock::now();
	    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	    std::cout << "Initialize LSH took " << elapsed.count() << " milliseconds." << std::endl;



	    Sgd *_sgd = new Sgd(_Algo, train_data, table_data, train_label, test_data, test_label, trainNum, testNum, width, conv, lr, epoch, decayrate,reg, minibatch, srpratio, K, L, outputFile);
		_sgd->LSDUpdate(adagrad);

	// }
	}

	else if (type==2){


		
		for (K=4; K<5; K++){
		LSH::_rangePow = K;
		if (LSH::_rangePow >= 32)
			cout << "Range of Hash Values Too Big" << endl;

		auto start = std::chrono::steady_clock::now();
		LSH *_Algo = new LSH(K, L);
	    auto end = std::chrono::steady_clock::now();
	    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	    std::cout << "Initialize LSH took " << elapsed.count() << " milliseconds." << std::endl;


	    Sgd *_sgd = new Sgd(_Algo, train_data, table_data, train_label, test_data, test_label, trainNum, testNum, width, conv, lr, epoch, decayrate,reg, minibatch, srpratio, K, L, outputFile);
	    _sgd->SGDUpdate(adagrad);
	    _sgd = new Sgd(_Algo, train_data, table_data, train_label, test_data, test_label, trainNum, testNum, width, conv, lr, epoch, decayrate,reg, minibatch, srpratio, K, L, outputFile);

		_sgd->LSDUpdate(adagrad);
	}
	}
}


int main(int argc, char* argv[])
{
	// std::time_t start, end;
	// long delta = 0;
	// start = std::time(NULL);


	volatile int i = 0; // "volatile" is to ask compiler not to optimize the loop away.
    auto start = std::chrono::steady_clock::now();

    LshSgd(argc, argv);


    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "It took me " << elapsed.count() << " milliseconds." << std::endl;

}
