from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

class ConvLayer():
    def __init__(self, learning_rate, img, filters):
        self.learning_rate = learning_rate
        self.img = img # n, 3, w, h
        self.filters = filters # f, 3, k, k


    def forwardBatch(self, images):
        self.images = images
        patches = sliding_window_view(
            images, 
            (self.filters.shape[2],self.filters.shape[3]), 
            (1, 2)
        )
        patches = patches.transpose(1, 2, 0, 3, 4)
        
        activationMap = np.sum(
            patches[None, ...] * self.filters[:, None, None, :, :, :],
            axis=(3, 4, 5)
        )

        return activationMap
    
    def backwardBatch(self, dOutput):
        _, kH, kW = self.filters.shape
        _, outH, outW = dOutput.shape

        dInput = np.zeros((self.img.shape))
        dFilters = np.zeros((self.filters.shape))  
        
        patches = sliding_window_view(
            dOutput, 
            (self.filters.shape[2],self.filters.shape[3]), 
            (1, 2)
        )
        patches = patches[None, ...]   
        dOutput = dOutput[:, :, :, None, None, None]  
 
        dFilters = np.sum(
            patches * dOutput,
            axis=(1, 2)
        )

        filters = self.filters[:, None, None, :, :, :] 
        contrib = dOutput * filters                     # (F, H, W, C, kH, kW)

        for i in range(kH):
            for j in range(kW):
                dInput[:, i:i+outH, j:j+outW] += np.sum(
                    contrib[:, :, :, :, i, j],
                    axis=0
                )
        
        self.dFilters = dFilters
        return dInput

    # def backward(self, dOutput):
    #     _, kH, kW = self.filters.shape
    #     outH = dOutput.shape[1]
    #     outW = dOutput.shape[2]
    #     dInput = np.zeros((self.img.shape))
    #     dFilters = np.zeros((self.filters.shape))

    #     for a in range(dFilters.shape[0]):
    #         for i in range(outH):
    #             for j in range(outW):
                    
    #                 patch = self.img[:, i:i+kH, j:j+kW]
    #                 dFilters[a] += patch * dOutput[a, i, j] 
    #                 dInput[:, i:i+kH, j:j+kW] += self.filters[a] * dOutput[a, i, j]
                
    #     self.dFilters = dFilters
    #     return dInput
    
    def update(self):
        self.filters -= self.learning_rate * self.dFilters
        
class ReLU():
    def __init__(self):
        pass
    
    def forward(self, matrix):
        self.matrix = matrix
        return np.maximum(matrix, 0)
    
    def backward(self, dA):
        return dA*(self.matrix > 0)
    
class MaxPool():
    def __init__(self, spanx, spany):
        self.spanx = spanx
        self.spany = spany

    def forward(self, matrix):
        self.matrix = matrix
        output = np.zeros((matrix.shape[0], matrix.shape[1]//self.spanx, matrix.shape[2]//self.spany))
        for h in range(0, matrix.shape[0]):
            for i in range(0, matrix.shape[1], self.spanx):
                for j in range(0, matrix.shape[2], self.spany):
                    output[h, i//self.spanx, j//self.spany] = np.max(matrix[h, i:i+self.spanx, j:j+self.spany])
        return output

    def backward(self, matrix, dA):
        output = np.zeros(matrix.shape)
        for h in range(0, matrix.shape[0]):
            for i in range(0, matrix.shape[1] - self.spanx + 1, self.spanx):
                for j in range(0, matrix.shape[2] - self.spany + 1, self.spany):
                    idx = np.argmax(matrix[h, i:i+self.spanx, j:j+self.spany])

                    row, col = np.unravel_index(idx, (self.spanx, self.spany))
                    output[h, i+ row, j+col] = dA[h, i//self.spanx, j//self.spany] 
        return output

class Dense():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(128, 15) * 0.01
        self.biases = np.zeros(15)
    
    def forward(self, matrix):
        flat = matrix.flatten()
        self.input = flat
        return Softmax().forward(flat.dot(self.weights) + self.biases)

    def backward(self, pred, actual):
        error = self.Loss(pred, actual)

        dW = np.outer(self.input, error)  # shape: (n_features, n_classes)
        db = error  # shape: (n_classes,)
        
        # Step 3: gradient w.r.t input (for backprop to previous layer)
        dInput = error.dot(self.weights.T)  # shape: (n_features,)
        dInput = dInput.reshape(self.input.shape)  # reshape if needed
        
        # Store gradients for optimizer
        self.dW = dW
        self.db = db

    def Loss(pred, target):
        return -np.sum(np.log(pred+1e-9) * target)
    
    def update(self):
        self.weights -= self.learning_rate * self.dW
        self.biases -= self.learning_rate * self.db

class Softmax():
    def forward(self, x):
        e_x = np.exp(x - np.max(x))
        self.pred = e_x / np.sum(e_x)
        return self.pred

F = 16
C = 3
k = 3

filters1 = np.random.randn(16, C, k, k) * np.sqrt(2 / (C * k * k))
filters2 = np.random.randn(32, C, k, k) * np.sqrt(2 / (16 * k * k))
filters3 = np.random.randn(32, C, k, k) * np.sqrt(2 / (C * k * k))

convolution1 = ConvLayer(0.01, None, filters1)
# convolution2 = ConvLayer(0.01, imgs, filters2)
# convolution3 = ConvLayer(0.01, imgs, filters3)

r = ReLU()
pool = MaxPool(2, 2)

d = Dense(0.01)

def forward(img):
    x = convolution1.forward(img)
    x = r.forward(x)
    x = pool.forward(x)
    
    pred = d.forward(x)
    return pred, x


def backward(pred, target, pooled_output):
    # Loss gradient
    loss = d.loss(pred, target)

    # Dense layer
    d.backward(pred, target)
    dDense = d.dInput.reshape(pooled_output.shape)

    # Pool layer
    dPool = pool.backward(r.matrix, dDense)

    # ReLU
    dRelu = r.backward(dPool)

    # Conv
    convolution1.backward(dRelu)

epochs = 3

for i in range(epochs)