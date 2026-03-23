import numpy as np

def convolution(img, filter):
    padding = filter.shape[0] // 2
    outH = img.shape[1] - filter.shape[1] + 1
    outW = img.shape[2] - filter.shape[2] + 1
    output = np.zeros((1, outH, outW))

    for i in range(padding, img.shape[1] - padding):
        for j in range(padding, img.shape[2] - padding):
            leftI = i-padding
            rightI = i+padding+1
            
            leftJ = j-padding
            rightJ = j+padding+1
            
            output[0, i, j] = np.sum(img[:, leftI:rightI, leftJ:rightJ] * filter)
            
    return output

def multipleFilters(img, filters):
    activationMap = np.zeros((filters.shape[0], img.shape[1], img.shape[2]))
    for i in range(filters.shape[0]):
        activationMap[i] = convolution(img, filters[i])[0]
    return activationMap

def ReLU(matrix):
    return np.maximum(matrix, 0)

def maxPool(matrix, spanx, spany):
    output = np.zeros((matrix.shape[0], matrix.shape[1]//spanx, matrix.shape[2]//spany))
    for h in range(0, matrix.shape[0]):
        for i in range(0, matrix.shape[1]-1, spanx):
            for j in range(0, matrix.shape[2]-1, spany):
                output[h, i//spanx, j//spany] = np.max(matrix[h, i:i+spanx, j:j+spany])
    return output

def avgPool(matrix, spanx, spany):
    output = np.zeros((matrix.shape[0], matrix.shape[1]//spanx, matrix.shape[2]//spany))
    for h in range(0, matrix.shape[0]):
        for i in range(0, matrix.shape[1], spanx):
            for j in range(0, matrix.shape[2], spany):
                output[h, i//spanx, j//spany] = np.mean(matrix[h, i:i+spanx, j:j+spany])
    return output

def crp(img, filter):
    activation = multipleFilters(img, filter)
    activation = ReLU(activation)
    activation = maxPool(activation)
    return activation

def ReLU_backward(dA, matrix):
    return dA*(matrix > 0)

def max_pool_backward(matrix, spanx, spany, dA):
    output = np.zeros(matrix.shape)
    for h in range(0, matrix.shape[0]):
        for i in range(0, matrix.shape[1] - spanx + 1, spanx):
            for j in range(0, matrix.shape[2] - spany + 1, spany):
                idx = np.argmax(matrix[h, i:i+spanx, j:j+spany])

                row, col = np.unravel_index(idx, (spanx, spany))
                output[h, i+ row, j+col] = dA[h, i//spanx, j//spany] 
    return output

def convolution_backward(matrix, ):
    # return multipleFilters(matrix, gradient)
    dFilter = convolution(matrix, dOutput)
    dInput  = convolution(dOutput, flipped_filter)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / np.sum(e_x)



def final(matrix, weights, biases):
    flat = matrix.flatten()
    return softmax(flat.dot(weights) + biases)

def reverse_final(pred, actual):
    error = actual - pred
    

i = np.array([[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
             [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
             [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]])

j1 = np.array([[[[2,2,2], [2,2,2], [2,2,2]], [[2,2,2], [2,2,2], [2,2,2]], [[2,2,2], [2,2,2], [2,2,2]]], 
               [[[3,3,3], [2,2,2], [2,2,2]], [[2,2,2], [2,2,2], [2,2,2]], [[2,2,2], [2,2,2], [2,2,2]]]])

print(avgPool(convolution(i, j1[0][0]), 2, 2))