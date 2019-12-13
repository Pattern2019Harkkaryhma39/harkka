# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:29:02 2019

@author: eskotakku
"""

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from parameters import cleaned_dataset_train, generated_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from parameters import cleaned_dataset_train, cleaned_dataset_test
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


######
#
#   Place resized data in arrays
#   X y 
#
#
#
######

class_names =sorted(os.listdir(cleaned_dataset_train))



stats = {}
for item in class_names:
    stats[item] = 0
for root, dirs, files in os.walk(cleaned_dataset_train):
    
    label = root.split(os.sep)[-1]
    X = ([])    # Feature vectors will go here.
    y = ([])    # Class ids will go here.

    if label in ["Boat", "Caterpillar", "Limousine", "Cart", "Bus", "Van","Car", "Barge"]:
        continue
    
    print(label, " Start")
    
    all_images = len([name for name in files if name.endswith(".jpg")])
    for name in files:    
        if name.endswith(".jpg"): 
            # Load the image:
            img = plt.imread(root + os.sep + name)

            # Convert the data to float, and normalize:
            img = img.astype(np.float32)
            img /= 255
            
            # Append the feature vector to our list.
            X.append(img)  
            
            # Extract class name from the directory name:
            # print(root)
            label = root.split(os.sep)[-1]
            y.append(class_names.index(label))
            
            stats[label] += 1
            print("\r{0} {1}/{2}".format(label, str(stats[label]), str(all_images)), end='')
    
    if label in ["Boat"]:
        A = X[:len(X)//2]
        B = X[len(X)//2:]
        pickle.dump(np.array(A), open(generated_model + os.sep + label + '.XA', 'wb'))
        pickle.dump(np.array(B), open(generated_model + os.sep + label + '.XB', 'wb'))
        pickle.dump(np.array(y), open(generated_model + os.sep + label + '.y', 'wb'))
    else:
        pickle.dump(np.array(X), open(generated_model + os.sep + label + '.X', 'wb'))
        pickle.dump(np.array(y), open(generated_model + os.sep + label + '.y', 'wb'))
    print("\n", root, " End")