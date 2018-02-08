import numpy as np
import matplotlib.pyplot as plt
"""
input file has two columns: train_error, test_error
"""

def plot(inputfile1):
	data = np.genfromtxt(inputfile1)

	#iters = range(len(data))
	# iters1 = data[:,1]
	# train_error = data[:,2]
	# test_error = data[:,3]
	color = ['b', 'g', 'r', 'm', 'y', 'k', 'c', 'b', 'g', 'r', 'b', 'g', 'r', 'm', 'y', 'k', 'c', 'b', 'g', 'r']

	dic = sorted(set(data[:, 0]))
	epoch = set(data[:, 3])


	for i in range(0, len(dic)):
		cur = list(dic)[i]

		iters1 = data[data[:, 0]==cur,:]
		# iters1 = iters1[iters1[:, 1]==0,:]
		lsderror = []
		sgderror = []
		avg_grad_norm = []
		avg_lsd_norm = []
		avg_true_norm = []
		for j in range(len(epoch)):
			
			e = list(epoch)[j]

			avg_lsd = iters1[iters1[:, 3]==e, 4]
			avg_sgd = iters1[iters1[:, 3]==e, 5]
			lsd_norm = iters1[iters1[:, 3]==e, 6]
			sgd_norm = iters1[iters1[:, 3]==e, 7]
			true_norm = iters1[iters1[:, 3]==e, 8]

			lsderror.append(np.average(avg_lsd))
			sgderror.append(np.average(avg_sgd))
			avg_lsd_norm.append(np.average(lsd_norm))
			avg_grad_norm.append(np.average(sgd_norm))
			avg_true_norm.append(np.average(true_norm))

		plt.plot(list(epoch), avg_lsd_norm, linestyle=":", color=color[i], label='lsd norm epoch='+str(cur))
		plt.plot(list(epoch), avg_grad_norm, linestyle="-", color=color[i+1], label='sgd norm epoch=' +str(cur))
		# plt.plot(list(epoch), avg_true_norm, linestyle="-.", color=color[i+2], label='true norm')
		# plt.plot(list(epoch), lsderror, linestyle=":", color=color[i], label='LSD K='+ str(cur))
		# plt.plot(list(epoch), sgderror, color=color[i+1],label='SGD K='+ str(cur))

	# plt.yscale('log')
	legend = plt.legend(loc='upper right', shadow=True)
	# plt.ylim(0, 10e2)
	# plt.xlim(0, 200)
	plt.ylabel('Norm')
	# plt.xlabel('time (ms)')
	plt.xlabel('samples')
	plt.show()

# plot("blog_estimate")
# plot("slice_estimation")
plot("test_error")
# a = np.array((np.zeros(5),np.zeros(5)))
# b = np.array((np.ones(5), np.ones(5), np.ones(5)))
# print a
# print b
# print a+b[:1]