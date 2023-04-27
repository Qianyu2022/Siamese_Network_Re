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







