import numpy as np

def convolution(img, filter):
    padding = filter.shape[0] // 2
    output = np.zeros((1, img.shape[1], img.shape[2]))
    for i in range(padding, img.shape[1] - padding):
        for j in range(padding, img.shape[2] - padding):
            leftI = i-padding
            rightI = i+padding+1
            
            leftJ = j-padding
            rightJ = j+padding+1
            
            output[0, i, j] = np.sum(img[:, leftI:rightI, leftJ:rightJ] * filter)
            
    return output

i = np.array([[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
             [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
             [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]])

j = np.array([[[2,2,2], [2,2,2], [2,2,2]], [[2,2,2], [2,2,2], [2,2,2]], [[2,2,2], [2,2,2], [2,2,2]]])

print(convolution(i, j))