from sklearn import preprocessing
import numpy as np
import csv
from numpy import linalg as LA
import math

def preprocess(inputname, header, delimiterby, trainname):
    data = np.genfromtxt(inputname, dtype=float, delimiter=delimiterby)
    if header:
		data = np.delete(data,(0),axis=0)

    # data = np.delete(data, (0), axis=1)
    print data.shape
    print data[1, :]

    np.random.shuffle(data)
    train_num = data.shape[0]/2

    Xtrain = data[:train_num]
    Xtest = data[train_num:]

    # print "First standarizing"
    # scaler = preprocessing.StandardScaler().fit(Xtrain)
    # Xtrain = scaler.transform(Xtrain) 

    # print "Then normalizing"
    # Xtrain_table = preprocessing.normalize(Xtrain,norm='l2')

    # S = np.sqrt(LA.norm(Xtrain, 2, axis=1)*LA.norm(Xtrain[:,:-1], 2, axis=1))
    # Xtrain_input = (Xtrain.T / S).T


    # Xtest = scaler.transform(Xtest)

    # print "test norm"
    # print LA.norm(Xtrain_table[5, :], 2)


    # print "training mean"
    # print scaler.mean_[-1]
    # print scaler.scale_[-1]

    # tablewriter = csv.writer(open(trainname+".table", 'wb'), delimiter= " ")
    # trainwriter_x = csv.writer(open(trainname+"_x.learn", 'wb'), delimiter= " ")
    # trainwriter_y = csv.writer(open(trainname+"_y.learn", 'wb'), delimiter= " ")
    # testwriter_x = csv.writer(open(trainname+"_x.test", 'wb'), delimiter= " ")
    # testwriter_y = csv.writer(open(trainname+"_y.test",'wb'),delimiter=" ")

    # tablewriter.writerows(Xtrain_input)
    # trainwriter_x.writerows( Xtrain_input[:, :-1])
    # trainwriter_y.writerow(Xtrain_input[:, -1])
    # for i in Xtrain_input:
    # trainwriter_y.writerow([i[-1]])
    # testwriter_x.writerows(Xtest[:, :-1])
    # for i in Xtest:
    #     testwriter_y.writerow([i[-1]])

# preprocess("/Users/beidichen/Documents/2018sp/Ironman_LSD/data/slice_localization_data.csv",True,",","slice")
preprocess("/Users/beidichen/Documents/2018sp/Ironman_LSD/data/cup98LRN.txt",True,",","cup")