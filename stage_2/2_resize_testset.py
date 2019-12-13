# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:27:00 2019

@author: Skosko
"""
import os
import cv2
from parameters import original_dataset_test, cleaned_dataset_test


    
for root, dirs, files in os.walk(original_dataset_test):
    
    all_images = len([name for name in files if name.endswith(".jpg")])
    processed = 0
    
    for name in files:
        if name.endswith(".jpg"): 
            # Load the image:
            img = cv2.imread(original_dataset_test + os.sep + name, cv2.IMREAD_UNCHANGED) 
            # Resize it to the net input size:
            img = cv2.resize(img, (224,224))
            
            
            cv2.imwrite(cleaned_dataset_test + os.sep + name, img)
            
            processed += 1
            
            print("\r{0}/{1}".format(processed, str(all_images)), end='')