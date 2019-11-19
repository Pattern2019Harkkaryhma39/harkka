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

X = []    # Feature vectors will go here.
y = []    # Class ids will go here.

stats = {}
for item in class_names:
    stats[item] = 0
for root, dirs, files in os.walk(cleaned_dataset_train):
    label = root.split(os.sep)[-1]
    
    # If running this script crashes midway you can uncomment these to skip them when rerunning
    
    # if label in ['', 'Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar', 'Helicopter']: # , 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile']:
    #    continue
    
    print(label, " Start")
    
    all_images = len([name for name in files if name.endswith(".jpg")])
    X = []
    y = []
    for name in files:    
        if name.endswith(".jpg"): 
            # Load the image:
            img = plt.imread(root + os.sep + name)
            
            # Resize it to the net input size:
            # img = cv2.resize(img, (224,224))
            
            # Convert the data to float, and remove mean:
            img = img.astype(np.float32)
            img -= 128
            
            # Push the data through the model:
            x = model.predict(img[np.newaxis, ...], batch_size=8,  use_multiprocessing=False)[0]
            
            # And append the feature vector to our list.
            X.append(x)
            
            # Extract class name from the directory name:
            # print(root)
            label = root.split(os.sep)[-1]
            y.append(class_names.index(label))
            stats[label] += 1
            print("\r{0} {1}/{2}".format(label, str(stats[label]), str(all_images)), end='')
            
    pickle.dump(np.array(X), open(generated_model + os.sep + label + '.X', 'wb'))
    pickle.dump(np.array(y), open(generated_model + os.sep + label + '.y', 'wb'))
    print("\n", root, " End")