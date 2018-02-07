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

	for i in range(0, len(dic)):
		cur = list(dic)[i]
		iters1 = data[data[:, 0]==cur,2]
		train_error = data[data[:, 0]==cur,3]
		test_error = data[data[:, 0]==cur,4]
		if train:
			plt.plot(iters1, train_error,linewidth=3, label='LSD Training Loss', color='r')
		
			#plt.plot(iters1, train_error,linewidth=3, label='LSD+adaGrad Training Loss', color='r')
		else:
			plt.plot(iters1, test_error, linewidth=3, label='LSD Testing Loss', color='r')
			#plt.plot(iters1, test_error, linewidth=3, label='LSD+adaGrad Testing Loss', color='r')


	iters2 = data2[:,2]
	train_error2 = data2[:,3]
	test_error2 = data2[:,4]

	if train:
		plt.plot(iters2, train_error2, linewidth=3, label='SGD Training Loss', color='b')
	else:
		plt.plot(iters2, test_error2, linewidth=3, label='SGD Testing Loss', color='b')
	# plt.yscale('symlog')
	legend = plt.legend(loc='upper right', shadow=True, fontsize=18)
	# plt.ylim(0, 10e2)
	# plt.xlim(0, 20000)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel('Loss', fontsize=18)
	plt.xlabel('Time (ms)', fontsize=18)
	plt.title('Slice Testing Loss',fontsize=18)
	# plt.xlabel('epoch')
	plt.show()

plot("lsd", "sgd",False)
