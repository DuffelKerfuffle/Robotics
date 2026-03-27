import numpy as np
from CNN1 import Dense, loadData, forward, loadModel, conv1, conv2, conv3, d
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cars = {
    0: "Toyota Prius",
    1: "Ford Focus",
    2: "Ferrari f40",
    3: "BMW E30 cross spoke",
    4: "Porsche 911",
    5: "BMW m3",
    6: "Audi",
    7: "Mercedes Benz E-class",
    8: "Tesla Model 3",
    9: "Jaguar XJ9220"
}

loadModel()

testingImages, testingTarget = loadData("testing")

testPred = forward(testingImages)
testError = Dense.loss(testPred, testingTarget)
print(f"Test loss: {testError:.4f}")

testingTarget = np.argmax(testingTarget, axis=1)
testPredLabels = np.argmax(testPred, axis=1)

class_names = list(cars.values())

cm = confusion_matrix(testingTarget, testPredLabels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.xticks(rotation=45, ha='right')  # rotate and right-align
plt.title("Confusion Matrix")
plt.tight_layout() 
plt.show()

cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix (Normalized)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()