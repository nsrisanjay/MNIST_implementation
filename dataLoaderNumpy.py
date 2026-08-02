import numpy as np

# for images data
def DataIntoNumpyArray(path:str,headerLength:int):
    with open(path,"rb") as file:
        fileByteBuffer = file.read()
        file.close()
    headersByteBuffer = fileByteBuffer[:headerLength]
    headers = []
    index = 0
    while index<headerLength-1:
        data = headersByteBuffer[index:index+4]
        headerElement = int.from_bytes(data,byteorder="big")
        headers.append(headerElement)
        index += 4
    totalNumberOfImages = headers[1]
    imageXDim = headers[2]
    imageYDim = headers[3]
    imagesByteBuffer = fileByteBuffer[headerLength:]
    imagesElementsArray = np.frombuffer(imagesByteBuffer,dtype=np.uint8)
    print("array shape : ",imagesElementsArray.shape)
    # reshape to (images,imageSize)
    # imagesArray = imagesElementsArray.reshape(totalNumberOfImages,imageXDim*imageYDim)

    # better for CNNs
    imagesArray = imagesElementsArray.reshape(totalNumberOfImages,imageXDim,imageYDim)

    # normalise values
    imagesArray = imagesArray/255.0
    return imagesArray
# for labels data
def labelsIntoNumpyArray(path:str,headerLength:int):
    with open(path,"rb") as file:
        fileByteBuffer = file.read()
        file.close()
    headersByteBuffer = fileByteBuffer[:headerLength]
    headers = []
    index = 0
    while index<headerLength-1:
        data = headersByteBuffer[index:index+4]
        headerElement = int.from_bytes(data,byteorder="big")
        headers.append(headerElement)
        index += 4
    numberOfLabels = headers[1]
    labelByteBuffer = fileByteBuffer[headerLength:]
    labelsElementsArray = np.frombuffer(labelByteBuffer,dtype=np.int8)
    print("array shape : ",labelsElementsArray.shape)
    return labelsElementsArray


# call these in other
# imagesArray = DataIntoNumpyArray("./dataset/train-images.idx3-ubyte",16)
# labelsArray = labelsIntoNumpyArray("./dataset/train-labels.idx1-ubyte",8)