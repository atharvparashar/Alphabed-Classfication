#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
# from skimage.feature import hog
import cv2
import pickle
import joblib


# In[7]:


print("dataSet Information")
# Gọi dữ liệu từ dataset kaggle
dataset = pd.read_csv('E:\CONTANT/A_Z Handwritten Data.csv').astype('float32')

dataset.info()


# In[4]:


dataset = dataset.sample(frac = 1)


X = dataset.drop('0',axis = 1)[:300000]
y = dataset['0'][:300000]


# In[6]:


print("shape of data")
print("X Shape : ",X.shape)
print("y Shape : ",y.shape)


# In[8]:


print("The label format representing numbers from (0 to 25 using 26 letters )")
print(sorted(y.unique()))
print(y.unique().shape)


# In[9]:


print("Displays 100 images of its letters and labels ")
alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
numbers_to_display = 100
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(20,20))
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X.iloc[i].values.reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(y.copy().map(alphabets_mapper).iloc[i])
plt.show()


# In[10]:


print("quantity of each label")
alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 
y_copy = y.copy().map(alphabets_mapper).sort_index()

label_size = y_copy.value_counts(sort=False)
print(label_size)
label_size.plot.barh(figsize=(10,10))
plt.xlabel("Number of elements ")
plt.ylabel("Alphabets")
plt.grid()
plt.show()


# In[12]:


X = np.array(X,dtype=np.float32)
y = np.array(y,dtype=np.float32)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

print("shape of data sets : ")
print("X_train : ",X_train.shape)
print("y_train : ",y_train.shape)
print("X_test : ",X_test.shape)
print("y_test : ",y_test.shape)


# In[13]:


X_train = X_train.reshape(X_train.shape[0],28,28)
X_test = X_test.reshape(X_test.shape[0],28,28)
print("X_train : ",X_train.shape)
print("X_test : ",X_test.shape)


# In[14]:


img = X_test[10].astype(np.uint8)
img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY )
# Find contours in the image
ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
plt.imshow(im_th)


# In[15]:


boxes = []
for c in ctrs:
    (x, y, w, h) = cv2.boundingRect(c)
    boxes.append([x,y, x+w,y+h])

boxes = np.asarray(boxes)
left, top = np.min(boxes, axis=0)[:2]
right, bottom = np.max(boxes, axis=0)[2:]
left,right,top, bottom


# In[16]:


plt.imshow(im_th[top-1:bottom+1,left-1:right+1])


# In[17]:


roi = im_th[top-1 : bottom+1,left-1 : right+1]
# Resize the image
roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
roi = cv2.dilate(roi, (3, 3))
plt.imshow(roi)


# In[18]:


list_ft = []
list_train_remove = []
i = -1
for img in X_train :
    i += 1
    img = img.astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Use GaussianBlur
    im_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    # Find contours in the image
    ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tìm khung chứa chữ số
    try:
        boxes = []
        for c in ctrs:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x,y, x+w,y+h])
        boxes = np.asarray(boxes)
        left, top = np.min(boxes, axis=0)[:2]
        right, bottom = np.max(boxes, axis=0)[2:]

        roi = im_th[top-1:bottom+1,left-1:right+1]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        
        roi = roi.reshape(-1)
        list_ft.append(roi)
    except Exception as e:
        list_train_remove.append(i)
        print(i,end=" - ")
features = np.array(list_ft, 'float32')


# In[19]:


features.shape


# In[20]:


labels = []
for i in range(y_train.shape[0]):
    if i not in list_train_remove :
        labels.append(y_train[i])
labels = np.array(labels)
print(labels[:10])
labels.shape


# In[21]:


alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

# Hiển thị 100 ảnh chữ số và label của nó 
numbers_to_display = 100
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(20,20))
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(features[i].reshape(28,28))
    plt.xlabel(pd.Series(labels).map(alphabets_mapper).iloc[i])
plt.show()


# In[22]:


alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
alphabets_mapper[2]


# In[23]:


features = features / 255
np.min(features), np.max(features)


# In[ ]:


model_svm_rbf = SVC(kernel='rbf')
model_svm_rbf.fit(features, labels)
print("Training succesfully")


# In[ ]:


list_X_test = []
list_test_remove = []
i = -1
for img in X_test :
    i += 1
    img = img.astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Use GaussianBlur
    im_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    # Find contours in the image
    ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    try:
        boxes = []
        for c in ctrs:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x,y, x+w,y+h])
        boxes = np.asarray(boxes)
        left, top = np.min(boxes, axis=0)[:2]
        right, bottom = np.max(boxes, axis=0)[2:]

        roi = im_th[top-1:bottom+1,left-1:right+1]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        
        roi = roi.reshape(-1)
        list_X_test.append(roi)
    except Exception as e:
        list_test_remove.append(i)
        print(i,end=" - ")
features_test = np.array(list_X_test, 'float32')


# In[ ]:


labels_test = []
for i in range(y_test.shape[0]):
    if i not in list_test_remove :
        labels_test.append(y_test[i])
labels_test = np.array(labels_test)
print(labels_test[:10])
print(features_test.shape)
print(labels_test.shape)


# In[ ]:


features_test = features_test / 255



y_pred_rbf = model_svm_rbf.predict(features_test)


print("accuracy :", accuracy_score(y_true=labels_test, y_pred=y_pred_rbf), "\n")

# matrix
print("accuracy matrix : ")
print(confusion_matrix(y_true=labels_test, y_pred=y_pred_rbf))


print("\naccuracy by class: ")
print(classification_report(labels_test, y_pred_rbf))


# In[ ]:


numbers_to_display = 100
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(20,20))
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i].reshape(28,28))
    plt.xlabel("Predict : "+str(pd.Series(y_pred_rbf).map(alphabets_mapper).iloc[i]))
plt.show()


# In[ ]:


filename = 'E:\CONTANT/working/model_svm_rbf.sav'
pickle.dump(model_svm_rbf, open(filename, 'wb'))


# In[ ]:


class TranningModelML():
    def __init__(self,features,labels,model):
        self.features = features
        self.labels = labels
        if model == "SVM":
            self.model = SVC(kernel='rbf')
        elif model == "KNN":
            self.model = KNeighborsClassifier(n_neighbors=80)
        elif model == "DecisionTree":
            self.model = DecisionTreeClassifier()
        elif model == "RandomForest":
            self.model = RandomForestClassifier()
        else :
            print("Không có mô hình đào tạo")
            exit()
            
        self.features_test = None
        self.labels_test = None
    
    def training(self):
        self.model.fit(self.features, self.labels)
        print("Score of model : ",model_svm_rbf.score(self.features, self.labels))
        
    def _process_data_X_test(self,X_test):
        # Xử lý features test
        list_X_test = []
        list_test_remove = []
        i = -1
        for img in X_test :
            i += 1
            img = img.astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Use GaussianBlur
            im_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

            # Threshold the image
            ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
            # Find contours in the image
            ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            
            try:
                boxes = []
                for c in ctrs:
                    (x, y, w, h) = cv2.boundingRect(c)
                    boxes.append([x,y, x+w,y+h])
                boxes = np.asarray(boxes)
                left, top = np.min(boxes, axis=0)[:2]
                right, bottom = np.max(boxes, axis=0)[2:]

                roi = im_th[top-1:bottom+1,left-1:right+1]
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (3, 3))

                roi = roi.reshape(-1)
                list_X_test.append(roi)
            except Exception as e:
                list_test_remove.append(i)
        features_test = np.array(list_X_test, 'float32')
        
        return features_test,list_test_remove
    
    def _process_data_testing(self,X_test,y_test):
        features_test,list_test_remove = self._process_data_X_test(X_test)
        
        # Xử lý label test
        labels_test = []
        for i in range(y_test.shape[0]):
            if i not in list_test_remove :
                labels_test.append(y_test[i])
        labels_test = np.array(labels_test)
        
        return features_test,labels_test
        
        
    def predict(self,X_test,y_test):
        features_test,labels_test = self._process_data_testing(X_test,y_test)
        
        self.features_test = features_test
        self.labels_test = labels_test
        
    
        features_test = features_test / 255

         
        y_pred_rbf = model_svm_rbf.predict(features_test)
        
        return y_pred_rbf
    
    def evaluate(self,y_pred_rbf):
        
        print("accuracy:", accuracy_score(y_true=self.labels_test, y_pred=y_pred_rbf), "\n")

        # matrix
        print("accuracy matrix : ")
        print(confusion_matrix(y_true=self.labels_test, y_pred=y_pred_rbf))

        
        print("\naccuracy by class: ")
        print(classification_report(self.labels_test, y_pred_rbf))
    
    def save(self,filename = ""):
        if filename == "":
            filename == 'E:\CONTANT/working/model/model.sav'
        pickle.dump(model_svm_rbf, open(filename, 'wb'))
        print("Save succesfully")


# In[ ]:




