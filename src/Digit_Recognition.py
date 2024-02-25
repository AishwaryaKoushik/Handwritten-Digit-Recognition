#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARIES AND DATASET

# In[88]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_digits
import pylab as pl  #To analyse a sample image
import seaborn as sns
from sklearn import metrics


# In[89]:


digits=load_digits()
digits.data.shape


# ## Analyse a Sample Image

# In[90]:


pl.gray()
pl.matshow(digits.images[250])
pl.show()


# ## Analyze Image Pixel

# Each element represents the pixel of our grayscale image.The value ranges from 0 to 255 for an 8 bit pixel

# In[91]:


digits.images[250]


# In[92]:


images_and_labels= list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5))
for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3,5,index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('%i' % label)


# ## Classifier Model-Support Vector Machine

# In[93]:


#from sklearn import svm
from sklearn import svm
from sklearn.model_selection import train_test_split


# In[94]:


x_train, x_test, y_train, y_test= train_test_split(x,y)
print(x_train.shape)
print(x_test.shape)


# # Define support Vector Classifier model- Linear Kernel

# In[95]:


model_linear=svm.SVC(kernel='linear', degree=3, gamma='scale')
model_linear.fit(x_train, y_train)
#prediction
y_pred=model_linear.predict(x_test)


# In[96]:


#EVALUATING THE MODEL ACCURACY


# In[97]:


score=model_linear.score(x_test, y_test)
print(score)


# In[98]:


i=250
pl.gray()
pl.matshow(digits.images[i])
pl.show()
classifier.predict(x[[i]])


# Define support Vector Classifier model-with RBF Kernel

# In[99]:


model_RBF=svm.SVC(kernel='rbf', degree=3, gamma='scale')
model_RBF.fit(x_train, y_train)
#prediction
predictions=model_RBF.predict(x_test)
score=model_RBF.score(x_test, y_test)
print(score)


# In[100]:


cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# In[101]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[102]:


i=250
pl.gray()
pl.matshow(digits.images[i])
pl.show()
classifier.predict(x[[i]])


# Check for prediction from dataset

# ## Classifier Model-KNN

# In[103]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[104]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)


# In[105]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[106]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
score=accuracy_score(y_test, y_pred)
print(score)


# In[107]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[108]:


i=250
pl.gray()
pl.matshow(digits.images[i])
pl.show()
classifier.predict(x[[i]])


# In[109]:


#Classifier Model- Random Forest
import random
from sklearn import ensemble

#Define variables
n_samples=len(digits.images)
x=digits.images.reshape((n_samples, -1))
y=digits.target

#Create random indices
sample_index=random.sample(range(len(x)),int(len(x)/5)) #20-80 split
valid_index=[i for i in range(len(x)) if i not in sample_index]

#Sample and Validation images
x_train=[x[i] for i in sample_index]
y_train=[x[i] for i in valid_index]

#Sample and Validation targets
x_test=[y[i] for i in sample_index]
y_test=[y[i] for i in valid_index]

#Using the Random Forest Classifier
classifier=ensemble.RandomForestClassifier()

#Fit model with sample data
classifier.fit(x_train, x_test)

#Predict Validation data
accuracy=classifier.score(y_train, y_test)
print('Random Tree Classifier:\n')
print("Accuracy\t" + str(accuracy))


#Random Tree Classifier:


i=250

pl.gray()
pl.matshow(digits.images[i])
pl.show()
classifier.predict(x[[i]])


# In[ ]:




