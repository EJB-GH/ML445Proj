import numpy as np
import torch
import os
#import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
#from sklearn.metrics import confusion_matrix, accuracy_score

'''
The dataset chosen is not divided nicely into sets for us
so I have to manually divide the sets up with ratios
the files themselves are 176x208 jpegs. I'd like to experiment
and leave then in their original ratio for now

The total size of the dataset is 6400 images.
[EB]
'''
device = torch.device('cuda')
torch.cuda.is_available()

#creating ratios for the division of the dataset
train_size, test_size, valid_size = 0.7, 0.15, 0.15

#batching will save us some time for computation
batch_size = 32

#training variables, subject to change based on results
epochs = None
lr = .08
momentum = .05

#ensure your path for this variable is the same as mine otherwise this wont work
path = 'Alzheimer_MRI_4_classes_dataset'

entire_set = datasets.ImageFolder(
    root=path,
    transform=transforms.ToTensor(),
)

#some diagnostic print outs to make sure youre loaded correctly
print(f"Total size of data loaded: {len(entire_set)}")
print(f"Classes found by ImageFolder call: {entire_set.classes}")
print(f"Classes in index: {entire_set.class_to_idx}")

#need to splice up the entire_set variable to these 3 sets
train_total = int(6400 * train_size)
test_total = int(6400 * test_size)
valid_total = 6400 - (train_total + test_total)

#diagnostic for the splicing math
print(f"{train_total} : {test_total} : {valid_total}")

#now divide the data using random_split
train_split, test_split, validation_split = random_split(entire_set, [train_total, test_total, valid_total])

train_data= DataLoader(train_split, batch_size=batch_size, shuffle=True)
test_set = DataLoader(test_split, batch_size=batch_size)
validation_set = DataLoader(validation_split, batch_size=batch_size)

print(f"Respective Sizes:Train-Test-Validation x32 for batch: {len(train_data)}, {len(test_set)}, {len(validation_set)}")

#helper relu function
def reLU(x):
    return torch.max(x,torch.tensor(0.0,device=device))


class Model():
    def __init__(self):

        #layer variables
        self.input_size = 36608 #176x208 much more complex than mnist so hidden is larger
        self.hidden_size = 256 #chosen randomly for the first test
        self.output_size = 4 #based on the classes available

        #layers themselves
        self.hl_w = torch.empty(self.hidden_size, self.input_size).to(device) #[256,36608]
        self.hl_b = torch.empty(1, self.hidden_size).to(device) #wanted to try 0 init vs the 1 we've used
        self.ol_w = torch.empty(self.output_size, self.hidden_size).to(device) #[4,256]
        self.ol_b = torch.empty(1, self.output_size).to(device)

        #doing initializations for the starting values
        torch.nn.init.uniform_(self.hl_w, -0.05, 0.05)  
        torch.nn.init.zeros_(self.hl_b)
        torch.nn.init.uniform_(self.ol_w, -0.05, 0.05)       
        torch.nn.init.zeros_(self.ol_b)

        #needed to create var to store the previous momentum updates
        self.ol_w_m = torch.zeros_like(self.ol_w).to(device)
        self.ol_b_m = torch.zeros_like(self.ol_b).to(device)
        self.hl_w_m = torch.zeros_like(self.hl_w).to(device)
        self.hl_b_m = torch.zeros_like(self.hl_b).to(device)
    
    def forward(self, x):
        x = x.view(x.shape[0],-1).to(device)
        hidden = reLU(x @ self.hl_w.T + self.hl_b)
        output = hidden @ self.ol_w.T + self.ol_b
        return output, hidden
    
    def backward(self,x, y, output, hidden):
        

#test model to verify build
model = Model()

print(f"Hidden Layer Shape: {model.hl_w.shape}")
print(f"Output Layer Shape: {model.ol_w.shape}")
print(f"Bias Setting: {model.ol_b}")