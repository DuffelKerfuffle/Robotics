from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps
import os, random, shutil

class ConvLayer():
    def __init__(self, learning_rate, images, filters):
        self.learning_rate = learning_rate
        self.images = images # n, 3, w, h
        self.filters = filters # f, 3, k, k

    def forward(self, images):
        self.images = images
        patches = sliding_window_view(
            images, 
            (self.filters.shape[2],self.filters.shape[3]), 
            (2, 3)
        )

        # activationMap: (F, N, outH, outW)
        activationMap = np.tensordot(
            self.filters,   # (F, C, kH, kW)
            patches,        # (N, C, outH, outW, kH, kW)
            axes=([1,2,3], [1,4,5])
        )
        # tensordot gives (F, N, outH, outW) → swap to (N, F, outH, outW)
        return activationMap.transpose(1, 0, 2, 3)

    def backward(self, dOutput):
        _, _, kH, kW = self.filters.shape
        _, _, outH, outW = dOutput.shape  # ← unpack 4 dims

        dInput = np.zeros(self.images.shape)  # (N, C, inH, inW)

        # gets patches which are the size of filters, turns into a list of patches
        
        patches = sliding_window_view(
            self.images,
            (kH, kW),
            (2, 3)
        )

        # find dot product over gradient and patches, condensing image dimensions since we're creating filters
        dFilters = np.tensordot(
            dOutput,   # (N, F, outH, outW)
            patches,   # (N, C, outH, outW, kH, kW)
            axes=([0, 2, 3], [0, 2, 3])  # sum over N, outH, outW
        )

        
        for i in range(kH):
            for j in range(kW):
                # apply filters to output matrix, by applying weights at each index across channels and filters
                contrib = np.tensordot(dOutput, self.filters[:, :, i, j], axes=([1], [0]))
                dInput[:, :, i:i+outH, j:j+outW] += contrib.transpose(0, 3, 1, 2)

        self.dFilters = dFilters
        return dInput
    
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
        
        # get all patches
        windows = sliding_window_view(
            matrix, 
            (self.spanx, self.spany), 
            (2, 3)
        )
        
        # get every other patch, so no overlapping patches
        windows = windows[:, :, ::self.spanx, ::self.spany, :, :]    

        #find max of each patch to collapse them
        return np.max(windows, axis=(4, 5))

    def backward(self, dA):
        N, F, H, W = self.matrix.shape
        outH, outW = dA.shape[2], dA.shape[3]

        # get all patches
        windows = sliding_window_view(
            self.matrix, 
            (self.spanx, self.spany), 
            (2, 3)
        )

        # get only overlapping patches
        windows = windows[:, :, ::self.spanx, ::self.spany]

        # flatten each window to 1d, finds max for each window
        flat = windows.reshape(N, F, outH, outW, -1)
        idx = np.argmax(flat, axis=-1) 

        #convert flat index into row/column index
        rows, cols = np.unravel_index(idx, (self.spanx, self.spany))
        
        # get top left row of EACH window
        base_i = np.arange(outH)[:, None] * self.spanx  
        base_j = np.arange(outW)[None, :] * self.spany  

        # get absolute position from RELATIVE position of max
        abs_rows = rows + base_i  
        abs_cols = cols + base_j  

        # add max dA values at the correct rows and columns
        output = np.zeros(self.matrix.shape)
        n_idx = np.arange(N)[:, None, None, None]
        f_idx = np.arange(F)[None, :, None, None]
        np.add.at(output, (n_idx, f_idx, abs_rows, abs_cols), dA)

        return output

class Dense():
    def __init__(self, learning_rate, input_size, output_size):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)
    
    def forward(self, matrix):
        self.original_shape = matrix.shape 
        flat = matrix.reshape(matrix.shape[0], -1)
        self.input = flat
        return Softmax().forward(flat.dot(self.weights) + self.biases)

    def backward(self, pred, actual):
        error = (pred - actual) / pred.shape[0]
        self.dW = self.input.T @ error
        self.db = np.sum(error, axis=0)

        dInput_flat = error @ self.weights.T
        self.dInput = dInput_flat.reshape(self.original_shape) 

    @staticmethod
    def loss(pred, target):
        return -np.mean(np.sum(target * np.log(pred + 1e-9), axis=1))
    
    def update(self):
        self.weights -= self.learning_rate * self.dW
        self.biases -= self.learning_rate * self.db

class Softmax():
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.pred = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.pred

def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]
F = 16
k = 3

filters1 = np.random.randn(16, 1, k, k) * np.sqrt(2 / (1 * k * k))
filters2 = np.random.randn(32, 16, k, k) * np.sqrt(2 / (16 * k * k))
filters3 = np.random.randn(64, 32, k, k) * np.sqrt(2 / (32 * k * k))

conv1 = ConvLayer(0.001, None, filters1)
conv2 = ConvLayer(0.001, None, filters2)
conv3 = ConvLayer(0.001, None, filters3)

r1, r2, r3 = ReLU(), ReLU(), ReLU()
pool1, pool2, pool3 = MaxPool(2, 2), MaxPool(2, 2), MaxPool(2, 2)

d = Dense(0.001, 64 * 6 * 6, 10)

root_folder = Path("Dataset")
TARGET_SIZE = (64, 64) 
def initialise():
    for subfolder in root_folder.iterdir():
        if subfolder.is_dir():
            print(f"\nFolder: {subfolder.name}")
            
            numbers = list(range(1, 16))
            random.shuffle(numbers)
            for sub in subfolder.iterdir():
                for file in sub.iterdir():
                    if file.suffix.lower() == '.png':
                        with Image.open(file) as img:
                            
                            img = img.convert("L")
                            img = ImageOps.pad(img, (64, 64), color=(0))
                            img.save(file)
                            
                            for angle in [90, 180, 270]:
                                rotated = img.rotate(angle, expand=True)

                                new_name = file.stem + f"_{angle}.png"
                                new_path = file.with_name(new_name)

                                rotated.save(new_path)

def rearrange():        
    for subfolder in root_folder.iterdir():
        if subfolder.is_dir():
            os.makedirs(root_folder / subfolder.name / "training", exist_ok=True)
            os.makedirs(root_folder / subfolder.name / "validation", exist_ok=True)
            os.makedirs(root_folder / subfolder.name / "testing", exist_ok=True)

            fileList = []
            for file in subfolder.iterdir():
                if file.suffix.lower() == '.png':
                    fileList.append(file)
                
            random.shuffle(fileList)

            for i in range(10):
                shutil.move(str(fileList[i]), root_folder / subfolder.name / "training" / fileList[i].name)
            for i in range(10, 13):
                shutil.move(str(fileList[i]), root_folder / subfolder.name / "validation" / fileList[i].name)
            for i in range(13, 15):
                shutil.move(str(fileList[i]), root_folder / subfolder.name / "testing" / fileList[i].name)

def loadData(folder):
    inputs = []
    outputs = []

    for subfolder in root_folder.iterdir():
        if subfolder.is_dir():
            target = int(subfolder.name)
            sub = subfolder / folder
            for file in sub.iterdir():
                if file.is_file() and file.suffix.lower() == '.png':
                    with Image.open(file) as img:
                        img = img.convert("L")
                        inputs.append(np.array(img))
                    outputs.append(target-1)
    out = np.array(inputs)          # shape: (N, H, W)
    out = out[:, np.newaxis, :, :]  # shape: (N, 1, H, W) — add channel dimension
    out = out / 255.0
    
    return out, one_hot(np.array(outputs))

def forward(images):
    x = conv1.forward(images)
    x = r1.forward(x)
    x = pool1.forward(x)

    x = conv2.forward(x)
    x = r2.forward(x)
    x = pool2.forward(x)

    x = conv3.forward(x)
    x = r3.forward(x)
    x = pool3.forward(x)

    pred = d.forward(x)
    return pred

def backward(pred, target):
    d.backward(pred, target)
    dx = d.dInput  
    
    dx = pool3.backward(dx)
    dx = r3.backward(dx)
    dx = conv3.backward(dx)

    dx = pool2.backward(dx)
    dx = r2.backward(dx)
    dx = conv2.backward(dx)

    dx = pool1.backward(dx)
    dx = r1.backward(dx)
    dx = conv1.backward(dx)

    conv1.update()
    conv2.update()
    conv3.update()
    d.update()


if __name__ == "__main__":
    trainingImages, trainingTarget = loadData("training")
    testingImages, testingTarget = loadData("testing")
    validationImages, validationTarget = loadData("validation")

    cars = {
        1: "Toyota Prius",
        2: "Ford Focus",
        3: "Ferrari f40",
        4: "BMW E30 cross spoke",
        5: "Porsche 911",
        6: "BMW m3",
        7: "Audi",
        8: "Mercedes Benz E-class",
        9: "Tesla Model 3",
        10: "Jaguar XJ9220"
    }

    trainingList = []
    validationList = []

    maxepochs = 500
    i = 0
    count = 0
    while i < maxepochs:
        i += 1
        print("forward", i)
        trainingPred = forward(trainingImages)
        trainingError = Dense.loss(trainingPred, trainingTarget)

        trainingList.append(trainingError)

        print("backward", i)
        backward(trainingPred, trainingTarget)

        validationPred = forward(validationImages)
        validationError = Dense.loss(validationPred, validationTarget)
        print(f"Epoch {i+1}: train loss {trainingError:.4f}, val loss {validationError:.4f}")
        if(len(validationList)) > 0:
            if(validationList[-1] <= validationError):
                count += 1
            else:
                count = 0
        if(count == 10):
            break

        validationList.append(validationError)

    testPred = forward(testingImages)
    testError = Dense.loss(testPred, testingTarget)

    predicted_classes = np.argmax(testPred, axis=1)
    true_classes      = np.argmax(testingTarget, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)

    print(f"Test loss: {testError:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    plt.plot(trainingList, label="Train loss")
    plt.plot(validationList, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_curves.png")
    plt.show()

    with open("losses.txt", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (t, v) in enumerate(zip(trainingList, validationList)):
            f.write(f"{i+1},{t:.6f},{v:.6f}\n")

def saveModel(path="model"):
    np.save(f"{path}/filters1.npy", conv1.filters)
    np.save(f"{path}/filters2.npy", conv2.filters)
    np.save(f"{path}/filters3.npy", conv3.filters)
    np.save(f"{path}/weights.npy", d.weights)
    np.save(f"{path}/biases.npy", d.biases)
    print(f"Model saved to {path}/")

def loadModel(path="model"):
    conv1.filters = np.load(f"{path}/filters1.npy")
    conv2.filters = np.load(f"{path}/filters2.npy")
    conv3.filters = np.load(f"{path}/filters3.npy")
    d.weights     = np.load(f"{path}/weights.npy")
    d.biases      = np.load(f"{path}/biases.npy")
    print(f"Model loaded from {path}/")

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    saveModel() 