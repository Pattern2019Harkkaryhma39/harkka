# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:39:38 2019

@author: Skosko

This file contains paths used. 

Paths can be relative or absolute.

Make sure that these folders exists before starting
"""

original_dataset_train = r"C:\Work\sgndataset\train" # This folder contains classes like Car and Caterpillar

cleaned_dataset_train = r".\resized\train" # location where traininset will be saved after it is resized

original_dataset_test = r"C:\Work\sgndataset\testset" # This folder contain jpg files we need to process

cleaned_dataset_test = r".\resized\test" # This cleaned jpg files will be saved here

generated_model = r".\resized\model" # Folder where generated models are saved