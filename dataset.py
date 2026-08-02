import torch

class MNISTdataset:
    def __init__(self,imagesArray,labelsArray):
        self.imagesTensor = torch.from_numpy(imagesArray)
        # numpy has float 64 -> mapped to tensor float 64, but float 32 is used
        self.imagesTensor = self.imagesTensor.to(torch.float32)
        self.imagesTensor = self.imagesTensor.unsqueeze(1)
        # print("--------",self.imagesTensor.shape)
        self.labelsTensor = torch.from_numpy(labelsArray)
        # same change the labels type to int64(long) from uint8, for loss function calculation purposes
        self.labelsTensor = self.labelsTensor.to(torch.long)
    
    def __len__(self):
        return self.imagesTensor.shape[0]
    def __getitem__(self, key):
        # return a tuple
        return (self.imagesTensor[key],self.labelsTensor[key])

    