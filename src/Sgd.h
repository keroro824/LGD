#include <string>
#include "LSH.h"

class Sgd {
private:
	double **_train_data;
	double **_table_data;
	double **_test_data;
	double **_train_label;
	double **_test_label;
	int _trainNum;
	int _testNum;
	int _dim;
	double _convrate;
	double _lr;
	int _epoch;
	double *_wv;
	double _wvz;
	int _randomNumber;
	double _decayrate;
	double _reg;
	int _ratio;
	int _check;

	std::string _outputFile;
	LSH *_Algo;
	int _K;
	int _L;
	double *_adagrad;
	double *_gradient;
public:
	Sgd(LSH *Algo, double** train_data, double** table_data, double** train_label, double** test_data, double** test_label, int trainNum, int testNum, int dim, double conv, double lr, int epoch, double decayrate, double reg, int check, int srpratio, int K, int L, std::string outputFile);
	Sgd(double** train_data, double** table_data, double** train_label, double** test_data, double** test_label, int trainNum, int testNum, int dim, double conv, double lr, int epoch, double decayrate, double reg, int check, std::string outputFile);
	int SGDUpdate(int ada);
	int LSDUpdate(int ada);
	void randomData(int* index ,int size);
	double dotproduct(double* a, double* b, int start, int size);
	int predict(std::string outputFile, int time, int iter);
	void freeze(SignedRandomProjection *srp, int iter, int train);
	int GDUpdate(int ada);
}; 