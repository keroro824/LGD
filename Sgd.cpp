#include "Sgd.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <cmath>
#include <math.h>
#include <fstream>
#include "LSH.h"
#include "SignedRandomProjection.h"
#include <chrono>
#include <random>
#include <set>
// #include "boost/random/discrete_distribution.h"
#pragma once
using namespace std;

Sgd::Sgd(double** train_data, double** table_data, double** train_label, double** test_data, double** test_label, int trainNum, int testNum, int dim, double conv, double lr, int epoch, double decayrate, double reg, int check, std::string outputFile)
{

	// _Algo = Algo;
    _train_data = train_data;
    _table_data = table_data;
    _test_data = test_data;
    _train_label = train_label;
    _test_label = test_label;
	_trainNum = trainNum;
	_testNum = testNum;
	_dim = dim;
	_convrate = conv;
	_epoch = epoch;
	_lr = lr;
	_wv = new double[dim+1];
	// _wv[0] = 1.0;

    _decayrate = decayrate;
    _outputFile = outputFile;
    _reg = reg;
    _check = check;

    _adagrad = new double[_dim];

    for(int i=0; i< _dim;i++)
    {
        _adagrad[i] = 0.0000001;
    }

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,0.000001);
	for(int i=0; i< dim;i++)
	{
        double number = distribution(generator);
		_wv[i] = number;
	}
	_wv[dim] = -1.0;
}


Sgd::Sgd(LSH *Algo, double** train_data, double** table_data, double** train_label, double** test_data, double** test_label, int trainNum, int testNum, int dim, double conv, double lr, int epoch, double decayrate, double reg, int check, int srpratio, int K, int L, std::string outputFile)
{
    _Algo = Algo;
    _train_data = train_data;
    _table_data = table_data;
    _test_data = test_data;
    _train_label = train_label;
    _test_label = test_label;
	_trainNum = trainNum;
	_testNum = testNum;
	_dim = dim;
	_convrate = conv;
	_epoch = epoch;
	_lr = lr;
	_wv = new double[dim+1];
	// _wv[0] = 1.0;
    _decayrate = decayrate;
    _outputFile = outputFile;
    _reg = reg;
    _ratio = srpratio;
    _check = check;
    _K = K;
    _L = L;
    _adagrad = new double[_dim];
    _gradient = new double[_dim];
    for(int i=0; i< _dim;i++)
    {
        _adagrad[i] = 0.0000001;
    }

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,0.000001);
	for(int i=0; i< dim;i++)
	{
        double number = distribution(generator);
		_wv[i] = number;
	}
	_wv[dim] = -1.0;
}


void Sgd::randomData(int* index ,int size)
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


double Sgd::dotproduct(double* a, double* b, int start, int size)
{   
    double total = 0.0;
    for (int i=start; i< size; i++)
    {
        total += a[i]*b[i];
    }   
    return total;
}


int Sgd::predict(string outputFile, int time, int iter)
{
    cout << "predicting\n";
    ofstream myfile(outputFile,  ios::out | ios::app);

    double trainerror = 0.0;
    double testerror = 0.0;
    double p_res, tmp;

    for (int i = 0; i < _trainNum; i++) 
    {
        p_res = dotproduct(_train_data[i], _wv, 0, _dim);
        double cur_label = _train_label[i][0];
        trainerror += (cur_label - p_res) * (cur_label - p_res);
    }

    for (int i = 0; i < _testNum; i++) 
    {
        p_res = dotproduct(_test_data[i], _wv, 0, _dim);
        double cur_label = _test_label[i][0];
        
        testerror += (cur_label - p_res) * (cur_label - p_res);
    }
    myfile<<_K <<" "<<iter<<" "<<time<<" "<<trainerror<<" "<<testerror<<endl;

    return 0;
}


int Sgd::SGDUpdate(int ada)
{
	//initialize timer
	auto start_timer = std::chrono::steady_clock::now();
    auto end_timer = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer);
    int t = 0;
	int cur_iter = 0;
	bool useada = false;

	//prepare ids for shuffling data
	int* index = new int[_trainNum];
	for (int i = 0; i<_trainNum; i++) 
    {
        index[i] = i;
    } 

    //initial prediction
    predict(_outputFile, 0,cur_iter);

    while (cur_iter < _epoch) {    

    	//timing
    	start_timer = std::chrono::steady_clock::now();

	    double p_res, tmp;
	    //shuffle data
	    randomData(index, _trainNum);

	    //per iteration update theta
	    for (int i = 0; i < _trainNum; i++) {

			double* td = _train_data[index[i]];
            double cur_label = _train_label[index[i]][0];

            //compute gradient
            double tmp = dotproduct(td, _wv, 0, _dim) - cur_label;

	    	for (int j = 0; j < _dim; j++) {

	    		double gradient = tmp * td[j];
                if (ada){
                	if (useada){
                		gradient = gradient/sqrt(_adagrad[j]);
                	}
                	_adagrad[j]+= gradient*gradient;
                }

                _wv[j]-= _lr  *  gradient;
            }

            // give some warm up before using adagrad
            if ((i==_trainNum/4) & ada){
            	useada = true;
            }
	    }

	    //clock epoch time
	    end_timer = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer);
        t+=elapsed.count();

	    cur_iter++;
	    predict(_outputFile, t, cur_iter);
	}

	return 0;
}

void Sgd::freeze(SignedRandomProjection *srp, int iter, int train){
	ofstream myfile("gradient_log",  ios::out | ios::app);
    double* truegrad = new double[_dim];

    double* grad = new double[_trainNum];
    int sizz = 1;

	//first compute true sgd
    double avg_true = 0.0;
    double true_gradient_norm = 0.0;
    for (int n=0; n<_trainNum; n++){
    	double cur_norm = 0.0;
        for (int f=0; f<_dim; f++){
            double tmp_gradient = _train_data[n][f]*(dotproduct(_train_data[n], _wv, 0, _dim) - _train_label[n][0]);
            if (n==0){
                truegrad[f] = tmp_gradient;
            }else{

                truegrad[f] += tmp_gradient;
            }
            cur_norm+=tmp_gradient;
        }
        grad[n] = cur_norm;
    }
    for (int f=0; f<_dim; f++){
        true_gradient_norm+= truegrad[f]*truegrad[f]/_trainNum/_trainNum;
    }

    // double variance = 0.0;
    // for (int n=0; n<_trainNum; n++){
    // 	variance+= pow(grad[n]-true_gradient_norm/(_trainNum*_trainNum), 2);
    // }
    // variance = sqrt(variance/(_trainNum-1));
    // myfile<< "gradient "<< variance/true_gradient_norm << " true gradient norm "<< true_gradient_norm << " std" <<variance <<endl; 

    double normwv =0.0;
    for (int m=0; m<_dim+1; m++)
    {
        normwv+=_wv[m]*_wv[m];
    }

    for (int batch=0; batch< 1000; batch++){

        double avg_lsd = 0.0;
        double avg_sgd = 0.0;
        double* sgd_grad = new double[_dim];
    	double* lsd_grad = new double[_dim];


        // cout<<" truegrad "<<endl;

        // int** sample_batch = new int *[3];
        // for( int bit = 0; bit < 3; bit++ ) {
        //     sample_batch[bit] = new int[sizz*(batch+1)];
        // }
        // _Algo->sampleBatch(_wv, srp,  sizz*(batch+1), sample_batch, 1);

        double sgd_error = 0.0;
        double lsd_error = 0.0;

        // cout<<"batch "<< batch<<endl;

        for (int sam =0; sam< sizz*(batch+1); sam++){


        	double lsd_norm =0;
        	double sgd_norm =0;
        	            //the compute sgd
            
            int samid = rand()%(_trainNum);

            double* td = _train_data[samid];
            double cur_label = _train_label[samid][0];
            double thetax = dotproduct(td, _wv, 0, _dim)-cur_label;

            for (int f=0; f<_dim; f++){

                sgd_norm+= pow(thetax*td[f], 2);
                // sgd_error += pow(truegrad[f]/_trainNum - thetax*td[f] , 2);
                // if (sam==0){
                sgd_grad[f] = thetax*td[f];
            	// }else{
             //    	sgd_grad[f] += thetax*td[f]/(sizz*(batch+1));
                // }
               
                // cout<< "True: "<< truegrad[f]/_trainNum <<" estimated sgd : "<<_gradient[f]<<endl;
            }  
            sgd_error += 1- acos(dotproduct(sgd_grad, truegrad, 0, _dim)/_trainNum/sqrt(true_gradient_norm*sgd_norm))/3.141592653;

            int* sample_batch;
            sample_batch = _Algo->sample(_wv, srp,1);



            // samid = sample_batch[0][sam];
            // int samset = sample_batch[1][sam];
            // int samtable = sample_batch[2][sam];

            samid = sample_batch[0];
            int samset = sample_batch[1];
            int samtable = sample_batch[2];


            // cout<< "sample id: " <<samid <<"samset: "<< samset<< "samtable: "<<samtable<<endl;

            td = _train_data[samid];
            cur_label = _train_label[samid][0];
            thetax = dotproduct(td, _wv,0 , _dim)-cur_label;

            double cp_comp = dotproduct(_table_data[samid], _wv, 0, _dim) - _table_data[samid][_dim];


            double cp = 1- acos(cp_comp/sqrt(normwv))/3.141592653;
            double prob = (1 - pow((1 - pow(cp, _K)),samtable))*(1.0/samset);
            // prob = 1.0/_trainNum;

            //first compute lsd

            for (int f=0; f<_dim; f++){
            	
            	double gradient = thetax*td[f];
                // lsd_error += pow(truegrad[f]/_trainNum - gradient/_trainNum/prob, 2);
                // cout<< "True: "<< truegrad[f]/_trainNum <<" estimated lsd: "<<_gradient[f]<<endl;
             //    if (sam==0){
             //    	lsd_grad[f] = gradient/(sizz*(batch+1))/_trainNum/prob;
            	// }else{
            	lsd_grad[f]= gradient;
            	// }
                lsd_norm += pow(gradient, 2);

            }
            lsd_error += 1- acos(dotproduct(lsd_grad, truegrad, 0, _dim)/_trainNum/sqrt(true_gradient_norm*lsd_norm))/3.141592653;
            avg_sgd += sqrt(sgd_norm);
            avg_lsd += sqrt(lsd_norm);            

        }
                
            // for (int f=0; f<_dim; f++){
            //     // lsd_error += pow(truegrad[f]/_trainNum -lsd_grad[f] , 2);
            //     // sgd_error += pow(truegrad[f]/_trainNum -sgd_grad[f] , 2);
            //     avg_lsd += pow(lsd_grad[f], 2);
            //     avg_sgd += pow(sgd_grad[f], 2);
            // }

			// lsd_error += 1- acos(dotproduct(lsd_grad, truegrad, 0, _dim)/_trainNum/sqrt(true_gradient_norm*avg_lsd))/3.141592653;
			// sgd_error += 1- acos(dotproduct(sgd_grad, truegrad, 0, _dim)/_trainNum/sqrt(true_gradient_norm*avg_sgd))/3.141592653;
       
        myfile<< _K<< " "<< iter <<" "<< train<< " "<< batch << " "<< lsd_error/(sizz*(batch+1)) << " "<< sgd_error/(sizz*(batch+1)) <<" "<< avg_lsd/(sizz*(batch+1)) << " "<< avg_sgd/(sizz*(batch+1))  << " "<< sqrt(true_gradient_norm) <<endl;

    }
}


int Sgd::LSDUpdate(int ada)
{
	auto start_timer = std::chrono::steady_clock::now();
    auto end_timer = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer);
    int t = 0;
	int cur_iter = 0;
	bool useada = false;

	//insert data to hashtables
	SignedRandomProjection *srp = new SignedRandomProjection(_dim+1, _K*_L, _ratio);
	cout << "Creating "<< _K << "hashes, " << _L <<"tables, " << _ratio <<"ratio" <<endl;
	for (int i = 0; i<_trainNum; i++) 
    {
    	int * cur_hash = srp->getHash(_table_data[i], 1);
        _Algo->add(cur_hash, i);
    } 
    // _Algo->count_unique();

    //initial prediction
    predict(_outputFile, 0,cur_iter);

    while (cur_iter < _epoch) {    

    	// std::set<int> query; 
    	//timing
    	start_timer = std::chrono::steady_clock::now();

	    //per iteration update theta
	    for (int i = 0; i < _trainNum; i++) {


	    	// if ((cur_iter%5==0) & (i%(_trainNum/4)==0)){
	    	// if ((cur_iter==0) & (i==(_trainNum/20))){
	    	// 	freeze(srp, cur_iter, i);
	    	// }
	    	

	    	//query data out
			int* sample = _Algo->sample(_wv, srp, 1);
            
            int sampleid = sample[0];
            int subset = sample[1];
            int table = sample[2];

            // query.insert(sampleid);

			double* td = _train_data[sampleid];
            double cur_label = _train_label[sampleid][0];

            //compute gradient


            double cp_comp =0.0;
            double tmp =0.0;

            double normwv =0.0;
            for (int m=0; m<_dim; m++)
            {
            	double th = _wv[m];
                normwv+= th*th;
                cp_comp+=_table_data[sampleid][m]*th;
                tmp+= td[m]*th;
            }
            tmp -= cur_label;
            cp_comp -= _table_data[sampleid][_dim];
            normwv += 1;

            double cp = 1- acos(cp_comp/sqrt(normwv))/3.141592653;
            double prob = (1 - pow((1 - pow(cp, _K)),table))*(1.0/subset);

            // if ((cp!=cp) &(normwv==normwv)){
            	// cout<< normwv <<" "<< prob<<" " << cp<<endl;
            // 	double nor = 0.0;
            // 	            for (int m=0; m<_dim+1; m++)
            // {
            //     nor+=_table_data[sampleid][m]*_table_data[sampleid][m];
            // }
            // cout<< nor<<endl;
            // }

	    	for (int j = 0; j < _dim; j++) {

	    		double gradient = tmp * td[j];
                if (ada){
                	if (useada){
                		gradient = gradient/sqrt(_adagrad[j]);
                	}
                	_adagrad[j]+= gradient*gradient;
                }

                _wv[j]-= _lr  *  gradient/prob/_trainNum;
            }

            // give some warm up before using adagrad
            if ((i==_trainNum/4) & ada){
            	useada = true;
            }

	    }
	    // cout << "query size" <<query.size()<<endl;
	    //clock epoch time
	    end_timer = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer);
        t+=elapsed.count();

	    cur_iter++;
	    predict(_outputFile, t, cur_iter);


	}

	return 0;
}


int Sgd::GDUpdate(int ada)
{
	//initialize timer
	auto start_timer = std::chrono::steady_clock::now();
    auto end_timer = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer);
    int t = 0;
	int cur_iter = 0;
	bool useada = false;

	//prepare ids for shuffling data 

    //initial prediction
    predict(_outputFile, 0,cur_iter);

    while (cur_iter < _epoch) {    

    	//timing
    	start_timer = std::chrono::steady_clock::now();


	    //per iteration update theta
	    for (int i = 0; i < 100; i++) {


	    	double* truegrad = new double[_dim];

	    	for (int n=0; n<_trainNum; n++){
		        for (int f=0; f<_dim; f++){
		            double tmp_gradient = _train_data[n][f]*(dotproduct(_train_data[n], _wv, 0, _dim) - _train_label[n][0]);
		            
		            if (n==0){
		                truegrad[f] = 1.0/_trainNum*tmp_gradient;
		            }else{

		                truegrad[f] += 1.0/_trainNum*tmp_gradient;
		            }
		        }
		    }

            //compute gradient

	    	for (int j = 0; j < _dim; j++) {
                _wv[j]-= _lr  *  truegrad[j];
            }
	    }

	    //clock epoch time
	    end_timer = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer);
        t+=elapsed.count();

	    cur_iter++;
	    predict(_outputFile, t, cur_iter);
	}

	return 0;
}










