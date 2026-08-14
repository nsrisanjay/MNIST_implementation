from dataLoaderNumpy import DataIntoNumpyArray, labelsIntoNumpyArray
from dataset import MNISTdataset
import numpy as np
from torch.utils.data import DataLoader
import torch
from model import CNN


TRAIN_IMAGES_PATH = "./dataset/train-images.idx3-ubyte"
TRAIN_LABELS_PATH = "./dataset/train-labels.idx1-ubyte"
TEST_IMAGES_PATH = "./dataset/t10k-images.idx3-ubyte" 
TEST_LABELS_PATH = "./dataset/t10k-labels.idx1-ubyte"

IMAGES_HEADER_LENGTH = 16
LABELS_HEADER_LENGTH = 8
EPOCHS = 10
LEARNING_RATE = 0.01

imagesArray = DataIntoNumpyArray(TRAIN_IMAGES_PATH,IMAGES_HEADER_LENGTH)
labelsArray = labelsIntoNumpyArray(TRAIN_LABELS_PATH,LABELS_HEADER_LENGTH)

#create dataset
Dataset = MNISTdataset(imagesArray,labelsArray)
print(len(Dataset))

dataLoader = DataLoader(Dataset,shuffle=True,batch_size=64)

# for batch_no, (images,label) in enumerate(dataLoader):
    # print(batch_no, ":" ,images.shape,images.dtype,label.shape,label.dtype)
model = CNN()
lossFunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

model.train()

for epoch in range(EPOCHS):
    lossPerEpoch = 0
    for batch_no ,(image,label) in enumerate(dataLoader):
        # forward pass
        predictedLabel = model(image)
        # compute Loss
        loss = lossFunction(predictedLabel,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossPerEpoch += loss.item()
    print(lossPerEpoch/len(dataLoader))

# TEST THE MODEL

testImagesData = DataIntoNumpyArray(TEST_IMAGES_PATH,IMAGES_HEADER_LENGTH)
testLabelsData = labelsIntoNumpyArray(TEST_LABELS_PATH,LABELS_HEADER_LENGTH)

testDataset = MNISTdataset(testImagesData,testLabelsData)


model.eval()

with torch.no_grad():
    outputs = model(testDataset.imagesTensor)
    predicted = torch.argmax(outputs,dim=1)
    correctPredictions = (predicted == testDataset.labelsTensor).sum().item()
    totalPredictions = len(testDataset)
    accuracy = (correctPredictions / totalPredictions) * 100

print(f"Test Accuracy: {accuracy}%")




