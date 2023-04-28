'''
Plot figures for project report
@Author: qiz19014
'''
# distribution for letters in the training set

import numpy as np
import matplotlib.pyplot as plt

# # dataset storing folder
# dataset_folder = '/Users/qianyu/Library/CloudStorage/OneDrive-UniversityofConnecticut/Course/CSE 5819/Paper Review/Omniglot Dataset/changed/train'

# dataset storing folder
dataset_folder = '/Users/qianyu/Library/CloudStorage/OneDrive-UniversityofConnecticut/Course/CSE 5819/Paper Review/Omniglot Dataset/changed/test'


# Read the folders to see how many different names excluding the number of the folders are there
import os
folder_names = os.listdir(dataset_folder)
folder_names = [name for name in folder_names if not name.startswith('_')]

# get rid of the number in the end of the name str
folder_names = [name[:-2] for name in folder_names]

# get the unique names and the number of each name
unique_names, counts = np.unique(folder_names, return_counts=True)

# plot the distribution
# the x-axis is the number from 1 to 25
# the y-axis is the number of images for each letter

plt.figure(figsize=(10, 5))
plt.bar(range(1,21), counts)
plt.xticks(rotation=90)
plt.xlabel('Letters')
plt.ylabel('Number of images')
plt.title('Distribution of letters in the test set')
plt.show()

# plot the train loss and train accuracy
# the x-axis is the number of epochs
# the y-axis is the loss or accuracy
# 90000 training pairs results
# train_loss = [0.341, 0.190, 0.151, 0.127, 0.112, 0.104, 0.093, 0.086,
#               0.082, 0.080, 0.074, 0.074, 0.068, 0.069, 0.063, 0.065,
#               0.062, 0.063, 0.059, 0.058, 0.057, 0.057, 0.057, 0.056,
#               0.053, 0.053, 0.054, 0.052, 0.052, 0.052, 0.051, 0.049]
# train_acc = [63.125, 68.125, 77.188, 76.875, 75.625, 84.062, 79.062, 81.250,
#              80.312, 83.125, 80.625, 82.188, 85.312, 86.875, 85.312, 86.875,
#              86.312, 88.750, 85.625, 87.500, 88.125, 84.062, 88.438, 83.750,
#              85.312, 86.250, 89.375, 88.750, 85.312, 86.875, 85.938, 87.812]
# 30000 training pairs results
train_acc = [27.5, 45.3125 ,53.4375, 57.5, 60.625, 63.75, 67.8125, 70.625
            ,72.1875, 68.75, 78.4375, 76.875, 77.1875, 75.9375, 77.5, 77.5
            ,78.4375, 76.25, 77.1875, 77.5, 81.5625, 76.5625, 81.875, 78.4375
            ,84.0625, 84.6875, 85, 84.0625, 80.625, 85, 84.0625, 83.125]

train_loss = [0.567, 0.297, 0.182, 0.158, 0.109, 0.112, 0.184, 0.120,
              0.093, 0.209, 0.200, 0.039, 0.174, 0.069, 0.068, 0.060,
              0.110, 0.070, 0.037, 0.090, 0.069, 0.057, 0.099, 0.124,
              0.103, 0.077, 0.074]


# plot the train loss and train accuracy
# the x-axis is the number of epochs
# the y-axis is the loss or accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, 28), train_loss)
plt.xticks(range(1, 28))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train loss')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, 33), train_acc)
plt.xticks(range(1, 33))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Val accuracy')
plt.show()









