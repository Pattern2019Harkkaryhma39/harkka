# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:29:02 2019

@author: eskotakku
"""

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import pickle

from parameters import cleaned_dataset_train, generated_model, cleaned_dataset_test

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# 2
class_names =sorted(os.listdir(cleaned_dataset_train))
print(class_names)


# 3

base_model = tf.keras.applications.mobilenet.MobileNet(input_shape = (224,224,3),include_top = False)

in_tensor = base_model.inputs[0]# Grab the input of base model
out_tensor = base_model.outputs[0]# Grab the output of base model

# Add an average pooling layer (averaging each of the 1024 channels):
out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)


# Define the full model by the endpoints.
model = tf.keras.models.Model(inputs  = [in_tensor],outputs = [out_tensor])

# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam') # optimizer = 'sgd'


# 4

# Find all image files in the data directory.

X =  [] # np.array() # Feature vectors will go here.
y =  [] # np.array([]) # Class ids will go here.
stats = {}
for item in class_names:
    
    
    l_X = pickle.load(open(generated_model + os.sep + item + '.X', 'rb'))
    l_y = pickle.load(open(generated_model + os.sep + item + '.y', 'rb'))
     
    # X = np.expand_dims((X, l_X))
    #for item in l_X:
    X.extend(l_X)
    # for item in l_y:
    y.extend(l_y)
    # y = np.vstack((y, l_y))
    
X = np.array(X)
y = np.array(y)       
    
print('X len:' , len(X))
print('y len:' , len(y))


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=12451)

# 5 a 
# Linear discriminant analysis classifier.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_acc = accuracy_score(lda.predict(X_test), y_test)
print('LinearDiscriminantAnalysis', lda_acc)
largest = lda
largest_val = lda_acc
print('LinearDiscriminantAnalysis is default best')

# 5 b 
# Support Vector machine linear
from sklearn.svm import SVC

svc_l = SVC(kernel='linear', gamma='auto')
svc_l.fit(X_train, y_train)
svc_l_acc = accuracy_score(svc_l.predict(X_test), y_test)
print('Linear Support Vector machine', svc_l_acc)

if svc_l_acc > largest_val:
    largest = svc_l
    largest_val =svc_l_acc
    print('Linear SVC is new best')
    


# 5 c 
#Support vector machine rbf

svc_r = SVC(kernel='rbf', gamma='auto')
svc_r.fit(X_train, y_train)
svc_r_acc = accuracy_score(svc_r.predict(X_test), y_test)

print('RBF Support Vector machine', svc_r_acc)

if svc_r_acc > largest_val:
    largest = svc_r
    largest_val = svc_r_acc
    print('RBF SVC is new best')
    
# 5 d
# Logistic regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
lr.fit(X_train, y_train)
lr_acc = accuracy_score(lr.predict(X_test), y_test)

print('Logistic regression', lr_acc)

if lr_acc > largest_val:
    largest = lr
    largest_val = lr_acc
    print('Logistic regression is new best')

# 5 e
# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0)
rf.fit(X_train, y_train)

rf_acc =  accuracy_score(rf.predict(X_test), y_test)

print('Random forest', rf_acc)

if rf_acc > largest_val:
    largest = rf
    largest_val = rf_acc
    print('Random forest is new best')


# print("In X", X[0])


# print("In trainset", X_train[0])
    

with open("submission.csv", "w") as fp:
    fp.write("Id,Category\n")
            # 3. predict class using the sklearn model
            # 4. convert class id to name (label = class_names[class_index])

    for root, dirs, files in os.walk(cleaned_dataset_test):
        for name in files:
            print(name)
            if name.endswith(".jpg"):                 
                # 1. load image and resize
                img = plt.imread(cleaned_dataset_test + os.sep + name)
                # img = cv2.resize(img, (224,224))
                
                # 2. vectorize using the net
                # Convert the data to float, and remove mean:
                img = img.astype(np.float32)
                img -= 128

                x = model.predict(img[np.newaxis, ...])[0]
                
                # print("Test set",img[np.newaxis, ...])
                # img[np.newaxis, ...]
                # 3. predict class using the sklearn model
                i = largest.predict([x])[0]
                
                print(i)
                # 4. convert class id to name (label = class_names[class_index])
                label = class_names[i]
                # print(i)
                # print(name.replace('.jpg', ''))
                fp.write("%s,%s\n" % (name.replace('.jpg', ''), label))
                # Push the data through the model:
                # x = model.predict(img[np.newaxis, ...])[0]