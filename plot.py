import numpy as np
import matplotlib.pyplot as plt

"""
input file has two columns: train_error, test_error
"""

def plot(inputfile1, inputfile2, train=True):
	data = np.genfromtxt(inputfile1)
	data2 = np.genfromtxt(inputfile2)

	#iters = range(len(data))
	# iters1 = data[:,1]
	# train_error = data[:,2]
	# test_error = data[:,3]


	dic = set(data[:, 0])

	for i in range(0, len(dic)-2):
		cur = list(dic)[i]
		iters1 = data[data[:, 0]==cur,2]
		train_error = data[data[:, 0]==cur,3]
		test_error = data[data[:, 0]==cur,4]
		if train:
			plt.plot(iters1, train_error,  label='LSD K='+ str(cur)+ 'training error')
		else:
			plt.plot(iters1, test_error, label='LSD K='+ str(cur)+ 'testing error')


	iters2 = data2[:,2]
	train_error2 = data2[:,3]
	test_error2 = data2[:,4]

	if train:
		plt.plot(iters2, train_error2, label='plain SGD training error')
	else:
		plt.plot(iters2, test_error2, label='plain SGD testing error')
	# plt.yscale('symlog')
	legend = plt.legend(loc='upper right', shadow=True)
	# plt.ylim(0, 10e2)
	plt.xlim(0, 20000)
	plt.ylabel('error')
	plt.xlabel('time (ms)')
	# plt.xlabel('epoch')
	plt.show()

plot("UJI_lsd_ada", "UJI_sgd_ada",True)
