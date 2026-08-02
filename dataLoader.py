# data loader
def convertBytesToInteger(arr):
    numberArr = []
    for d in arr:
        numberArr.append(d)
    return numberArr

with open("./dataset/train-images.idx3-ubyte","rb") as file:
    fileContent = file.read()
    file.close()

headers = fileContent[0:16]
decodedHeader = []
index = 0
while index < 15:
    data = headers[index:index+4]
    number = int.from_bytes(data,byteorder="big")
    decodedHeader.append(number)
    index = index+4
totalImages = decodedHeader[1]
imageX = decodedHeader[2]
imageY = decodedHeader[3]
pixelsPerImage = imageX * imageY
images = []

init = 16
for i in range(0,totalImages):
    pixelArray = fileContent[init:init+pixelsPerImage]
    pixelArray = convertBytesToInteger(pixelArray)
    images.append(pixelArray)
    init += pixelsPerImage
###################################################################################################
with open("./dataset/train-labels.idx1-ubyte","rb") as file:
    fileContent2 = file.read()
    headersByteFormat = fileContent2[0:8]
    file.close()

decodedHeaderForLabels = []
index = 0
while index<7:
    data = headersByteFormat[index:index+4]
    number = int.from_bytes(data,byteorder="big")
    decodedHeaderForLabels.append(number)
    index += 4
# print(decodedHeaderForLabels)
NumberOfLabels = decodedHeaderForLabels[1]
init = 8
labelArray = []
for i in range(0,NumberOfLabels):
    data = fileContent2[init]
    labelArray.append(data)
    init += 1
print(labelArray[:10])
print(len(labelArray))


image = images[0]

for row in range(28):
    for col in range(28):
        pixel = image[row * 28 + col]
        print(f"{pixel:3}", end=" ")
    print()
print(labelArray[0])
# print(images[0])