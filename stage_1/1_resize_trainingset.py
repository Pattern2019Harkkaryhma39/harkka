# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:27:00 2019

@author: Skosko
"""

import os
import cv2
from parameters import original_dataset_train, cleaned_dataset_train

class_names =sorted(os.listdir(original_dataset_train))

stats = {}

for item in class_names:
    if not os.path.exists(cleaned_dataset_train + os.sep + item):
        os.makedirs(cleaned_dataset_train + os.sep + item)
    stats[item] = 0
    
for root, dirs, files in os.walk(original_dataset_train):
    for name in files:
        
        all_images = len([name for name in files if name.endswith(".jpg")])
        
        if name.endswith(".jpg"): 
            # Load the image:
            img = cv2.imread(root + os.sep + name, cv2.IMREAD_UNCHANGED) 
            # Resize it to the net input size:
            img = cv2.resize(img, (224,224))
            
            
            label = root.split(os.sep)[-1]
            
            
            cv2.imwrite(cleaned_dataset_train + os.sep + label + os.sep + name, img)
            
            stats[label] += 1
            
            print("\r{0} {1}/{2}".format(label, str(stats[label]), str(all_images)), end='')
        print(label, " done")