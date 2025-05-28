import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score

'''
The dataset chosen is not divided nicely into sets for us
so I have to manually divide the sets up with ratios
the files themselves are 176x208 jpegs. I'd like to experiment
and leave then in their original ratio for now

The total size of the dataset is 6400 images.
[EB]
'''

#creating ratios for the division of the dataset
train_size, test_size, valid_size = 0.7, 0.15, 0.15

#batching will save us some time for computation
batch_size = 32

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