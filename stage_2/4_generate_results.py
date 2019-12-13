#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from parameters import cleaned_dataset_train, cleaned_dataset_test, generated_model
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

######
#
#   Place resized data in arrays
#   X y 
#
#
#
######

class_names =sorted(os.listdir(cleaned_dataset_train))
X =  [] # np.array() # Feature vectors will go here.
y =  [] # np.array([]) # Class ids will go here.

for item in class_names:
    if item in ["Boat"]:
        l_X = pickle.load(open(generated_model + os.sep + item + '.XA', 'rb'))
        X.extend(l_X)
        l_X = pickle.load(open(generated_model + os.sep + item + '.XB', 'rb'))
        X.extend(l_X)
    else:
        l_X = pickle.load(open(generated_model + os.sep + item + '.X', 'rb'))
        X.extend(l_X)
    
    l_y = pickle.load(open(generated_model + os.sep + item + '.y', 'rb'))
     
    # X = np.expand_dims((X, l_X))
    # for item in l_y:
    y.extend(l_y)
    # y = np.vstack((y, l_y))
    
X = np.array(X)
y = np.array(y)  

print(X.shape)
print(y.shape)



# Split the data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)
 
# Convert class vectors to binary class matrices

batch_size = 4
num_classes = len(class_names)
epochs = 12

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)





#####
# RUN MOBILENET TO DATA
#
#
#
#####


print("mobilenet start")

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# input_shape=None, alpha=1.0, include_top=True, weights='imagenet', input_tensor=None, pooling=None, 

base_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(224,224,3),alpha = 0.25, include_top=False, classes=num_classes)

w = base_model.output

 

w = GlobalAveragePooling2D()(w)

w = Flatten()(w)

w = Dense(100, activation='relu')(w)

output = Dense(num_classes, activation='softmax')(w)

 

model = Model(inputs = [base_model.input], outputs = [output])


model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

aug = ImageDataGenerator(rotation_range=20, 
                         zoom_range=0.15,
                         width_shift_range=0.2, 
                         height_shift_range=0.2, 
                         shear_range=0.15,
                         horizontal_flip=True, 
                         fill_mode="nearest")
 
# Create callback feature for val_loss to reduce overfitting

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit_generator(aug.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    verbose=1,
                    steps_per_epoch=len(X_train) // batch_size,
                    validation_data=(X_test, y_test),
                    callbacks=[es_callback])

# Evaluate

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predict and create results


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
                
                # Convert the data to float and
                img = img.astype(np.float32)
                img /= 255

                x = model.predict(img[np.newaxis, ...])[0]
                x = np.argmax(x)

                print(x)
                
                # 4. convert class id to name (label = class_names[class_index])
                label = class_names[x]
                print(label)
                #print(i)
                # print(name.replace('.jpg', ''))
                fp.write("%s,%s\n" % (name.replace('.jpg', ''), label))
