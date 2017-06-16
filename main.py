from __future__ import division
from sklearn import svm
import numpy as np
import random
import pandas as pd
import torch
from torch.autograd import Variable
import math
import scipy as scip
from scipy import linalg
from scipy.spatial.distance import pdist, squareform

class loss_function(torch.nn.Module):

    def forward(self, loss):
        return torch.mean(loss)

##
#Note: Normalize data to avoid explosion or decay of numerical values, especially for calculating inverse
##

sigma = 1.0
noise = 0.00001

def accuracy(error):
    num = 0
    for item in error:
        if item == 0:
            num = num + 1
    return (num/len(error))*100

data = []

with open('splice2.data', 'r') as f:
    for line in f:
        array = []
        line = line.split(',')
        for item in line:
            item = item.strip()
            array.append(item)
        del array[1::2]
        if array[0] == 'IE':
            array[0] = 1.0
        if array[0] == 'EI':
            array[0] = 2.0
        if array[0] == 'N':
            array[0] = 3.0
        sequence = []
        for seq in array[1]:
            if seq == 'A':
                sequence.append(1.0/8.0)
            if seq == 'C':
                sequence.append(2.0/8.0)
            if seq == 'G':
                sequence.append(3.0/8.0)
            if seq == 'T':
                sequence.append(4.0/8.0)
            if seq == 'N':
                sequence.append(5.0/8.0)
            if seq == 'D':
                sequence.append(6.0/8.0)
            if seq == 'S':
                sequence.append(7.0/8.0)
            if seq == 'R':
                sequence.append(8.0/8.0)
        array[1] = sequence
        data.append(array[1] + [array[0]])

data = np.array(data)
np.random.shuffle(data)
data = pd.DataFrame(data)
print data
y = data[data.columns[60]]
del data[data.columns[60]]
x = data
x = np.array(x .as_matrix(), dtype=np.float64)
y = np.array(y.values, dtype=np.float64)

x_train = x[:2233]
y_train = y[:2233]
x_test = x[2233:]
y_test = y[2233:]


#SVM RBF Model
clf = svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)
ans = clf.predict(x_train)
error =  y_train - ans
print 'Sci-kit Learn SVM Vanilla Model: ', accuracy(error)


#SVM computed gram matrix
clf = svm.SVC(kernel='precomputed')
pairwise_dists = squareform(pdist(x_train, 'euclidean'))
for iteration in xrange(5):
    my_kernel = scip.exp(-(pairwise_dists ** 2) / sigma ** 2)
    k_inv = np.linalg.inv(my_kernel + noise*np.identity(len(my_kernel)))
    #er = 0.5*(np.dot(np.dot(np.dot(k_inv, y_train), y_train.T), k_inv) - k_inv)
    d_my_kernel = (2/ (sigma ** 3))*scip.exp(- (pairwise_dists ** 2) / sigma ** 2)
    er = 0.5*np.trace(np.dot(k_inv,d_my_kernel)) - 0.5*np.dot(np.dot(np.dot(np.dot(y_train.T,k_inv),d_my_kernel),k_inv),y_train)
    #print sigma
    sigma = sigma - er

my_kernel = scip.exp(-(squareform(pdist(x_test, 'euclidean')) ** 2) / sigma ** 2)
clf.fit(my_kernel, y_test)
ans = clf.predict(my_kernel)
error =  y_test - ans
print 'SVM trained by gradient descent: ', accuracy(error)
print 'Done'


# Deep Learning Model
inpt_train_x = torch.from_numpy(x_train)
inpt_train_x = inpt_train_x.float()
inpt_train_y = torch.from_numpy(y_train)
inpt_train_y = inpt_train_y.float()

inpt_train_x = Variable(inpt_train_x)
inpt_train_y = Variable(inpt_train_y, requires_grad=False)


# Vanilla Deep Learning Model
model = torch.nn.Sequential(
    torch.nn.Linear(60, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(20000):

    y_pred = model(inpt_train_x)
    loss = loss_fn(y_pred, inpt_train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


inpt_test_x = torch.from_numpy(x_test)
inpt_test_x = inpt_test_x.float()
inpt_test_x = Variable(inpt_test_x)

array = model.forward(inpt_test_x).data.numpy()
new_array = []
for index in array:
    if index[0] <= 1.5:
        new_array.append(1)
    if index[0] > 1.5 and index[0] <= 2.5:
        new_array.append(2)
    if index[0] > 2.5:
        new_array.append(3)
new_array = np.array(new_array)
error = y_test - new_array

deep_acc = accuracy(error)
print 'Vanilla Deep Learning Model Accuracy: ' ,deep_acc, '%'


# Deep Kernel Learning Model
model = torch.nn.Sequential(
    torch.nn.Linear(60, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,3)
)

loss_fn = loss_function()

learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(4):

    y_pred = model(inpt_train_x)
    
    g = y_pred.data.numpy()
    g_k = squareform(pdist(g, 'euclidean'))
    my_kernel = scip.exp(-(g_k ** 2) / sigma ** 2)
    k_inv = np.linalg.inv(my_kernel + noise*np.identity(len(my_kernel)))
    d_my_kernel = (2/ (sigma ** 3))*scip.exp(- (g_k ** 2) / sigma ** 2)
    e = 0.5*np.trace(np.dot(k_inv,d_my_kernel)) - 0.5*np.dot(np.dot(np.dot(np.dot(y_train.T,k_inv),d_my_kernel),k_inv),y_train)
    er = 0.5*(np.dot(np.dot(np.dot(k_inv, y_train), y_train.T), k_inv) - k_inv) 
    er = np.dot(er,g)
    sigma = sigma - e

    g_p = torch.from_numpy(np.matrix(er))
    g_p = g_p.float()
    g_p = Variable(g_p, requires_grad=True)

    loss = loss_fn(g_p)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


inpt_test_x = torch.from_numpy(x_test)
inpt_test_x = inpt_test_x.float()
inpt_test_x = Variable(inpt_test_x)

array = model.forward(inpt_test_x).data.numpy()
array = squareform(pdist(array, 'euclidean'))
array_kernel = scip.exp(-(array ** 2) / sigma ** 2)

clf = svm.SVC(kernel='precomputed')
clf.fit(array_kernel,y_test)

ans = clf.predict(y_test)
error =  y_test - ans
print 'Deep Kernel Learning Accuracy: ' ,accuracy(error)