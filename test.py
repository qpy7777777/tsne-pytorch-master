import numpy as np
import torch
# data = torch.rand(2,3)
# print(data)
# print(torch.sum(data))
# print(data / torch.sum(data))
# print(len(data[:,0]))
# print(len(data[:,1]))
def normalization(data):
    max_data = np.max(data)
    min_data = np.min(data)
    new_audio = (data-min_data)/(max_data-min_data)
    return new_audio
import numpy as np

##设置全部数据，不输出省略号
import sys

np.set_printoptions(threshold=sys.maxsize)
boxes=np.load('mel_384_train.npy')
np.savetxt('mel384.txt',boxes,fmt='%s',newline='\n')
X = np.loadtxt("mel384.txt")
X = normalization(X)
X = torch.Tensor(X)
print(X.shape)
print(torch.sum(X))
# np.savetxt('boxes.txt', boxes, fmt='%s', newline='\n')
print('---------------------boxes--------------------------')

